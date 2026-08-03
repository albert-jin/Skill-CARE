#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune SkillRouter-Reranker-0.6B with exact listwise cross-entropy.

Expected flat training fields:
    instruction_text
    skill_name
    skill_description
    skill_body
    label

Supported input formats:
    .jsonl / .json / .csv / .parquet

Important:
1. Rows are grouped by instruction_text. Each group is one reranking candidate list.
2. The recommended group size is 20, matching the SkillRouter paper.
3. At least one label=1 candidate must exist in every retained group.
4. The model score is the final-position logit difference:
       score(q, s) = logit("yes") - logit("no")
5. The listwise loss supports one or multiple positives:
       L = logsumexp(all_scores / tau) - logsumexp(positive_scores / tau)
   For a single positive, this is exactly standard listwise cross-entropy.
6. To fit a 32GB GPU, exact listwise gradients are computed with a two-pass method:
   first obtain all scores without gradients, then recompute candidate micro-batches
   and backpropagate the analytically exact gradient coefficients.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from src.common import (
    format_rerank_prompt,
    get_reranker_template_tokens,
    tokenize_reranker_text,
)


LOGGER = logging.getLogger("train_skillrouter_reranker")


@dataclass
class Candidate:
    skill_name: str
    skill_description: str
    skill_body: str
    label: int


@dataclass
class CandidateGroup:
    instruction_text: str
    candidates: List[Candidate]


@dataclass
class TrainConfig:
    model_path: str
    train_file: str
    output_dir: str
    max_length: int
    description_max_chars: int
    body_max_chars: int
    group_size: int
    candidate_micro_batch_size: int
    gradient_accumulation_groups: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    listwise_temperature: float
    val_ratio: float
    seed: int
    max_grad_norm: float
    num_workers: int
    use_flash_attention_2: bool
    trust_remote_code: bool
    filter_false_negatives: bool
    body_jaccard_threshold: float
    save_every_steps: int
    log_every_steps: int
    milestone_fractions: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listwise fine-tuning for SkillRouter-Reranker-0.6B"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/SkillRouter-Reranker-0.6B",
        help="Local base reranker directory.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Training data: jsonl/json/csv/parquet.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="models",
        help="The timestamped run directory is created below this directory.",
    )

    # Paper setting: reranker field caps 500/2000 chars and max length 4096.
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--description_max_chars", type=int, default=500)
    parser.add_argument("--body_max_chars", type=int, default=2000)
    parser.add_argument("--group_size", type=int, default=20)

    # 5090 32GB conservative defaults.
    parser.add_argument("--candidate_micro_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_groups", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--listwise_temperature", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--use_flash_attention_2",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--filter_false_negatives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Remove obvious false negatives by same-name/body trigram overlap. "
            "Keep this disabled for counterfactual training, because intended "
            "hard negatives usually share the positive skill name."
        ),
    )
    parser.add_argument("--body_jaccard_threshold", type=float, default=0.6)
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="Optional extra step checkpoints inside the run directory. Default 0 disables them.",
    )
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument(
        "--milestone_fractions",
        type=float,
        nargs="*",
        default=[0.4, 0.7],
        help=(
            "Overall optimizer-step progress fractions at which to save full "
            "model checkpoints as sibling directories, e.g. 0.4 0.7."
        ),
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(float(value) > 0)
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "yes", "y", "positive", "relevant"}:
        return 1
    if value_str in {"0", "false", "no", "n", "negative", "irrelevant"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def load_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        return rows

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("data", "rows", "instances", "records"):
                if isinstance(obj.get(key), list):
                    return obj[key]
        raise ValueError("JSON must be a list or contain data/rows/instances/records.")

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading parquet requires pandas and pyarrow.") from exc
        return pd.read_parquet(path).to_dict(orient="records")

    raise ValueError(f"Unsupported file extension: {suffix}")


def trigram_set(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[i : i + 3] for i in range(len(normalized) - 2)}


def trigram_jaccard(a: str, b: str) -> float:
    set_a = trigram_set(a)
    set_b = trigram_set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def filter_obvious_false_negatives(
    candidates: List[Candidate],
    body_jaccard_threshold: float,
) -> Tuple[List[Candidate], int]:
    positives = [c for c in candidates if c.label == 1]
    if not positives:
        return candidates, 0

    positive_names = {
        re.sub(r"\s+", " ", c.skill_name.lower()).strip()
        for c in positives
        if c.skill_name.strip()
    }

    retained: List[Candidate] = []
    removed = 0
    for candidate in candidates:
        if candidate.label == 1:
            retained.append(candidate)
            continue

        normalized_name = re.sub(
            r"\s+", " ", candidate.skill_name.lower()
        ).strip()
        same_name = bool(normalized_name and normalized_name in positive_names)
        high_body_overlap = any(
            trigram_jaccard(candidate.skill_body, positive.skill_body)
            > body_jaccard_threshold
            for positive in positives
            if candidate.skill_body and positive.skill_body
        )

        if same_name or high_body_overlap:
            removed += 1
        else:
            retained.append(candidate)

    return retained, removed


def build_groups(
    rows: Sequence[Dict[str, Any]],
    group_size: int,
    filter_false_negatives: bool,
    body_jaccard_threshold: float,
) -> List[CandidateGroup]:
    required = {
        "instruction_text",
        "skill_name",
        "skill_description",
        "skill_body",
        "label",
    }
    if not rows:
        raise ValueError("Training file is empty.")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    grouped: Dict[str, List[Candidate]] = defaultdict(list)
    for row_idx, row in enumerate(rows):
        query = normalize_text(row.get("instruction_text"))
        if not query:
            LOGGER.warning("Skip row %d: empty instruction_text", row_idx)
            continue
        grouped[query].append(
            Candidate(
                skill_name=normalize_text(row.get("skill_name")),
                skill_description=normalize_text(row.get("skill_description")),
                skill_body=normalize_text(row.get("skill_body")),
                label=parse_label(row.get("label")),
            )
        )

    output: List[CandidateGroup] = []
    skipped_no_positive = 0
    removed_false_negatives = 0

    for query, candidates in grouped.items():
        if filter_false_negatives:
            candidates, removed = filter_obvious_false_negatives(
                candidates, body_jaccard_threshold
            )
            removed_false_negatives += removed

        positives = [c for c in candidates if c.label == 1]
        negatives = [c for c in candidates if c.label == 0]
        if not positives:
            skipped_no_positive += 1
            continue

        # If a query has more than group_size candidates, create deterministic
        # groups while ensuring every group contains at least one positive.
        if group_size > 0 and len(candidates) > group_size:
            # Preserve all positives where possible. For the common single-positive
            # setup, split negatives into chunks of group_size - 1.
            if len(positives) >= group_size:
                LOGGER.warning(
                    "Query has %d positives >= group_size=%d; retaining all candidates "
                    "as one group to avoid dropping positives.",
                    len(positives),
                    group_size,
                )
                output.append(CandidateGroup(query, candidates))
                continue

            neg_chunk_size = max(1, group_size - len(positives))
            for start in range(0, len(negatives), neg_chunk_size):
                chunk = positives + negatives[start : start + neg_chunk_size]
                if len(chunk) >= 2:
                    output.append(CandidateGroup(query, chunk))
        else:
            if len(candidates) >= 2:
                output.append(CandidateGroup(query, candidates))

    LOGGER.info(
        "Built %d listwise groups from %d unique queries; "
        "skipped_no_positive=%d, removed_obvious_false_negatives=%d",
        len(output),
        len(grouped),
        skipped_no_positive,
        removed_false_negatives,
    )
    if not output:
        raise ValueError("No valid listwise groups were created.")
    return output


class GroupDataset(Dataset):
    def __init__(self, groups: Sequence[CandidateGroup]) -> None:
        self.groups = list(groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> CandidateGroup:
        return self.groups[index]


def identity_collate(items: List[CandidateGroup]) -> CandidateGroup:
    if len(items) != 1:
        raise ValueError("DataLoader batch_size must be 1 listwise group.")
    return items[0]


def split_groups(
    groups: List[CandidateGroup], val_ratio: float, seed: int
) -> Tuple[List[CandidateGroup], List[CandidateGroup]]:
    shuffled = groups.copy()
    random.Random(seed).shuffle(shuffled)

    if val_ratio <= 0 or len(shuffled) < 2:
        return shuffled, []

    val_size = max(1, int(round(len(shuffled) * val_ratio)))
    val_size = min(val_size, len(shuffled) - 1)
    return shuffled[val_size:], shuffled[:val_size]


def build_reranker_prompt(
    tokenizer: Any,
    query: str,
    candidate: Candidate,
    description_max_chars: int,
    body_max_chars: int,
) -> str:
    del tokenizer
    return format_rerank_prompt(
        candidate.skill_name,
        candidate.skill_description,
        candidate.skill_body,
        query,
        prompt_format="flat-full",
        desc_max=description_max_chars,
        body_max=body_max_chars,
    )


def find_answer_token_id(tokenizer: Any, word: str) -> int:
    candidates = [word, f" {word}", f"\n{word}"]
    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]

    token_id = tokenizer.convert_tokens_to_ids(word)
    if token_id is not None and token_id != tokenizer.unk_token_id:
        return int(token_id)

    raise ValueError(
        f"Could not find a single token id for {word!r}. "
        "Inspect the tokenizer vocabulary/chat template."
    )


def tokenize_prompts(
    tokenizer: Any,
    prompts: Sequence[str],
    max_length: int,
    prefix_tokens: Sequence[int],
    suffix_tokens: Sequence[int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    encoded_ids = [
        tokenize_reranker_text(
            prompt,
            tokenizer,
            list(prefix_tokens),
            list(suffix_tokens),
            max_length,
        )
        for prompt in prompts
    ]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch_max_len = max(len(ids) for ids in encoded_ids)
    padded: list[list[int]] = []
    attention_masks: list[list[int]] = []
    for ids in encoded_ids:
        pad_len = batch_max_len - len(ids)
        padded.append([pad_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))
    return {
        "input_ids": torch.tensor(padded, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
    }


def get_base_transformer(model: torch.nn.Module) -> torch.nn.Module:
    # Qwen3ForCausalLM exposes .model and .lm_head.
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise AttributeError(
            "Expected a causal LM with `.model` and `.lm_head` attributes."
        )
    return model.model


def score_prompts(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    max_length: int,
    yes_token_id: int,
    no_token_id: int,
    prefix_tokens: Sequence[int],
    suffix_tokens: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    inputs = tokenize_prompts(
        tokenizer,
        prompts,
        max_length,
        prefix_tokens,
        suffix_tokens,
        device,
    )
    base_model = get_base_transformer(model)
    outputs = base_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    last_hidden = outputs.last_hidden_state[:, -1, :]
    answer_logits = F.linear(
        last_hidden,
        model.lm_head.weight[[no_token_id, yes_token_id]],
    )
    # Raw logit difference, consistent with Qwen3 reranker scoring.
    return answer_logits[:, 1] - answer_logits[:, 0]


def score_group_in_chunks(
    model: torch.nn.Module,
    tokenizer: Any,
    group: CandidateGroup,
    config: TrainConfig,
    yes_token_id: int,
    no_token_id: int,
    prefix_tokens: Sequence[int],
    suffix_tokens: Sequence[int],
    device: torch.device,
    require_grad: bool,
) -> torch.Tensor:
    prompts = [
        build_reranker_prompt(
            tokenizer,
            group.instruction_text,
            candidate,
            config.description_max_chars,
            config.body_max_chars,
        )
        for candidate in group.candidates
    ]

    scores: List[torch.Tensor] = []
    context = torch.enable_grad if require_grad else torch.no_grad
    with context():
        for start in range(0, len(prompts), config.candidate_micro_batch_size):
            chunk = prompts[start : start + config.candidate_micro_batch_size]
            scores.append(
                score_prompts(
                    model,
                    tokenizer,
                    chunk,
                    config.max_length,
                    yes_token_id,
                    no_token_id,
                    prefix_tokens,
                    suffix_tokens,
                    device,
                )
            )
    return torch.cat(scores, dim=0)


def exact_listwise_loss_and_score_gradients(
    scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("listwise_temperature must be > 0.")
    positive_mask = labels.bool()
    if not torch.any(positive_mask):
        raise ValueError("Every group must contain at least one positive.")

    scaled = scores.float() / temperature
    loss = torch.logsumexp(scaled, dim=0) - torch.logsumexp(
        scaled[positive_mask], dim=0
    )

    p_all = torch.softmax(scaled, dim=0)
    p_pos = torch.zeros_like(p_all)
    p_pos[positive_mask] = torch.softmax(scaled[positive_mask], dim=0)

    # dL / d(score)
    grad_scores = (p_all - p_pos) / temperature
    return loss, grad_scores


def train_one_group_exact_two_pass(
    model: torch.nn.Module,
    tokenizer: Any,
    group: CandidateGroup,
    config: TrainConfig,
    yes_token_id: int,
    no_token_id: int,
    prefix_tokens: Sequence[int],
    suffix_tokens: Sequence[int],
    device: torch.device,
    loss_scale: float,
) -> Tuple[float, bool]:
    labels = torch.tensor(
        [c.label for c in group.candidates],
        device=device,
        dtype=torch.long,
    )

    # Pass 1: obtain the exact scores and exact listwise gradient coefficients.
    model.eval()
    with torch.no_grad():
        detached_scores = score_group_in_chunks(
            model,
            tokenizer,
            group,
            config,
            yes_token_id,
            no_token_id,
            prefix_tokens,
            suffix_tokens,
            device,
            require_grad=False,
        )
        loss, score_grads = exact_listwise_loss_and_score_gradients(
            detached_scores,
            labels,
            config.listwise_temperature,
        )

    # Pass 2: recompute small candidate chunks and inject dL/dscore.
    model.train()
    prompts = [
        build_reranker_prompt(
            tokenizer,
            group.instruction_text,
            candidate,
            config.description_max_chars,
            config.body_max_chars,
        )
        for candidate in group.candidates
    ]

    for start in range(0, len(prompts), config.candidate_micro_batch_size):
        end = min(start + config.candidate_micro_batch_size, len(prompts))
        chunk_scores = score_prompts(
            model,
            tokenizer,
            prompts[start:end],
            config.max_length,
            yes_token_id,
            no_token_id,
            prefix_tokens,
            suffix_tokens,
            device,
        )
        chunk_grads = score_grads[start:end].to(
            dtype=chunk_scores.dtype,
            device=chunk_scores.device,
        )
        surrogate = torch.sum(chunk_scores * chunk_grads.detach()) * loss_scale
        surrogate.backward()

    top_index = int(torch.argmax(detached_scores).item())
    hit1 = bool(labels[top_index].item() == 1)
    return float(loss.item()), hit1


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    groups: Sequence[CandidateGroup],
    config: TrainConfig,
    yes_token_id: int,
    no_token_id: int,
    prefix_tokens: Sequence[int],
    suffix_tokens: Sequence[int],
    device: torch.device,
) -> Dict[str, float]:
    if not groups:
        return {}

    model.eval()
    total_loss = 0.0
    total_hit1 = 0
    reciprocal_rank_sum = 0.0

    for group in groups:
        labels = torch.tensor(
            [c.label for c in group.candidates],
            device=device,
            dtype=torch.long,
        )
        scores = score_group_in_chunks(
            model,
            tokenizer,
            group,
            config,
            yes_token_id,
            no_token_id,
            prefix_tokens,
            suffix_tokens,
            device,
            require_grad=False,
        )
        loss, _ = exact_listwise_loss_and_score_gradients(
            scores,
            labels,
            config.listwise_temperature,
        )
        total_loss += float(loss.item())

        ranking = torch.argsort(scores, descending=True)
        ranked_labels = labels[ranking]
        positive_positions = torch.nonzero(
            ranked_labels == 1, as_tuple=False
        ).flatten()
        if len(positive_positions) > 0:
            first_rank = int(positive_positions[0].item()) + 1
            reciprocal_rank_sum += 1.0 / first_rank
            if first_rank == 1:
                total_hit1 += 1

    count = len(groups)
    return {
        "loss": total_loss / count,
        "hit@1": total_hit1 / count,
        "mrr": reciprocal_rank_sum / count,
    }


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    optimizer_step: int,
    epoch: int,
    metrics: Dict[str, float] | None = None,
) -> None:
    del optimizer_step, epoch, metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(output_dir)


def normalize_milestone_fractions(values: Sequence[float]) -> List[float]:
    normalized: list[float] = []
    for value in values:
        fraction = float(value)
        if fraction <= 0 or fraction >= 1:
            raise ValueError(
                f"milestone_fractions must be between 0 and 1, got {value!r}"
            )
        if fraction not in normalized:
            normalized.append(fraction)
    return sorted(normalized)


def format_milestone_fraction(fraction: float) -> str:
    return f"{fraction:g}"


def write_checkpoint_manifest(path: Path, entries: Sequence[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(list(entries), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d-%H%M%S")
    output_root = Path(args.output_root)
    run_name = f"Counterfactual-SkillRouter-Reranker-0.6B-{timestamp}"
    output_dir = output_root / run_name
    final_dir = output_dir / f"{run_name}-final"
    setup_logging(output_dir)
    set_seed(args.seed)
    milestone_fractions = normalize_milestone_fractions(args.milestone_fractions)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training script.")

    device = torch.device("cuda")
    model_path = Path(args.model_path)
    train_file = Path(args.train_file)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    config = TrainConfig(
        model_path=str(model_path),
        train_file=str(train_file),
        output_dir=str(output_dir),
        max_length=args.max_length,
        description_max_chars=args.description_max_chars,
        body_max_chars=args.body_max_chars,
        group_size=args.group_size,
        candidate_micro_batch_size=args.candidate_micro_batch_size,
        gradient_accumulation_groups=args.gradient_accumulation_groups,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        listwise_temperature=args.listwise_temperature,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        use_flash_attention_2=args.use_flash_attention_2,
        trust_remote_code=args.trust_remote_code,
        filter_false_negatives=args.filter_false_negatives,
        body_jaccard_threshold=args.body_jaccard_threshold,
        save_every_steps=args.save_every_steps,
        log_every_steps=args.log_every_steps,
        milestone_fractions=milestone_fractions,
    )

    with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    yes_token_id = find_answer_token_id(tokenizer, "yes")
    no_token_id = find_answer_token_id(tokenizer, "no")
    prefix_tokens, suffix_tokens = get_reranker_template_tokens(tokenizer)
    LOGGER.info(
        "Answer token ids: yes=%d, no=%d", yes_token_id, no_token_id
    )
    LOGGER.info(
        "Reranker template token lengths: prefix=%d, suffix=%d",
        len(prefix_tokens),
        len(suffix_tokens),
    )

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, **model_kwargs
        )
    except (ImportError, ValueError) as exc:
        if config.use_flash_attention_2:
            LOGGER.warning(
                "FlashAttention-2 unavailable (%s). Falling back to SDPA.", exc
            )
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                model_path, **model_kwargs
            )
        else:
            raise

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    LOGGER.info(
        "Model params: total=%s, trainable=%s",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    rows = load_rows(train_file)
    groups = build_groups(
        rows,
        group_size=config.group_size,
        filter_false_negatives=config.filter_false_negatives,
        body_jaccard_threshold=config.body_jaccard_threshold,
    )
    train_groups, val_groups = split_groups(
        groups, config.val_ratio, config.seed
    )
    LOGGER.info(
        "Train groups=%d, validation groups=%d",
        len(train_groups),
        len(val_groups),
    )

    train_loader = DataLoader(
        GroupDataset(train_groups),
        batch_size=1,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=identity_collate,
        pin_memory=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_groups
    )
    total_optimizer_steps = optimizer_steps_per_epoch * config.epochs
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    milestone_checkpoints = [
        {
            "fraction": fraction,
            "label": format_milestone_fraction(fraction),
            "step": max(1, math.ceil(total_optimizer_steps * fraction)),
            "output_dir": output_dir / f"{run_name}-{format_milestone_fraction(fraction)}",
        }
        for fraction in config.milestone_fractions
    ]

    LOGGER.info(
        "Training: epochs=%d, groups/epoch=%d, optimizer_steps=%d, "
        "candidate_micro_batch=%d, grad_accum_groups=%d, max_length=%d",
        config.epochs,
        len(train_loader),
        total_optimizer_steps,
        config.candidate_micro_batch_size,
        config.gradient_accumulation_groups,
        config.max_length,
    )
    total_train_groups = len(train_loader) * config.epochs
    LOGGER.info(
        "Progress units: total_train_groups=%d, optimizer_steps_per_epoch=%d, "
        "total_optimizer_steps=%d",
        total_train_groups,
        optimizer_steps_per_epoch,
        total_optimizer_steps,
    )
    if milestone_checkpoints:
        LOGGER.info(
            "Milestone checkpoints: %s",
            ", ".join(
                f"{item['label']}@step{item['step']}->{item['output_dir']}"
                for item in milestone_checkpoints
            ),
        )

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    best_metric = -float("inf")
    best_metrics: Dict[str, float] | None = None
    best_epoch = 0
    best_optimizer_step = 0
    saved_milestone_labels: set[str] = set()
    checkpoint_manifest: list[dict[str, Any]] = []
    checkpoint_manifest_path = output_dir / "checkpoint_manifest.json"

    with tqdm(
        total=total_train_groups,
        desc=f"Epoch 1/{config.epochs}",
        unit="group",
        dynamic_ncols=True,
    ) as train_progress:
        for epoch in range(1, config.epochs + 1):
            running_loss = 0.0
            running_hit = 0
            groups_since_log = 0

            train_progress.set_description(f"Epoch {epoch}/{config.epochs}")
            for group_idx, group in enumerate(train_loader, start=1):
                loss_scale = 1.0 / config.gradient_accumulation_groups
                loss_value, hit1 = train_one_group_exact_two_pass(
                    model,
                    tokenizer,
                    group,
                    config,
                    yes_token_id,
                    no_token_id,
                    prefix_tokens,
                    suffix_tokens,
                    device,
                    loss_scale=loss_scale,
                )
                running_loss += loss_value
                running_hit += int(hit1)
                groups_since_log += 1

                avg_loss = running_loss / max(1, groups_since_log)
                avg_hit = running_hit / max(1, groups_since_log)
                train_progress.set_postfix(
                    epoch=f"{epoch}/{config.epochs}",
                    group=f"{group_idx}/{len(train_loader)}",
                    opt_step=f"{optimizer_step}/{total_optimizer_steps}",
                    loss=f"{avg_loss:.4f}",
                    hit1=f"{avg_hit:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    refresh=False,
                )
                train_progress.update(1)

                should_step = (
                    group_idx % config.gradient_accumulation_groups == 0
                    or group_idx == len(train_loader)
                )
                if should_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                    train_progress.set_postfix(
                        epoch=f"{epoch}/{config.epochs}",
                        group=f"{group_idx}/{len(train_loader)}",
                        opt_step=f"{optimizer_step}/{total_optimizer_steps}",
                        loss=f"{avg_loss:.4f}",
                        hit1=f"{avg_hit:.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        grad_norm=f"{float(grad_norm):.2f}",
                        refresh=False,
                    )

                    for milestone in milestone_checkpoints:
                        label = str(milestone["label"])
                        if (
                            label not in saved_milestone_labels
                            and optimizer_step >= int(milestone["step"])
                        ):
                            milestone_dir = Path(milestone["output_dir"])
                            save_checkpoint(
                                model,
                                tokenizer,
                                milestone_dir,
                                optimizer_step,
                                epoch,
                                {
                                    "milestone_fraction": float(milestone["fraction"]),
                                    "milestone_step": optimizer_step,
                                    "total_optimizer_steps": total_optimizer_steps,
                                },
                            )
                            saved_milestone_labels.add(label)
                            checkpoint_manifest.append(
                                {
                                    "type": "milestone",
                                    "fraction": float(milestone["fraction"]),
                                    "epoch": epoch,
                                    "optimizer_step": optimizer_step,
                                    "total_optimizer_steps": total_optimizer_steps,
                                    "path": str(milestone_dir),
                                }
                            )
                            write_checkpoint_manifest(
                                checkpoint_manifest_path,
                                checkpoint_manifest,
                            )
                            LOGGER.info(
                                "Saved milestone checkpoint %.0f%% at step %d/%d: %s",
                                float(milestone["fraction"]) * 100,
                                optimizer_step,
                                total_optimizer_steps,
                                milestone_dir,
                            )

                    if (
                        config.log_every_steps > 0
                        and optimizer_step % config.log_every_steps == 0
                    ):
                        LOGGER.info(
                            "epoch=%d/%d step=%d/%d group=%d/%d "
                            "loss=%.6f hit@1=%.4f lr=%.3e grad_norm=%.4f "
                            "gpu_alloc=%.2fGB gpu_reserved=%.2fGB",
                            epoch,
                            config.epochs,
                            optimizer_step,
                            total_optimizer_steps,
                            group_idx,
                            len(train_loader),
                            avg_loss,
                            avg_hit,
                            scheduler.get_last_lr()[0],
                            float(grad_norm),
                            torch.cuda.memory_allocated() / 1024**3,
                            torch.cuda.memory_reserved() / 1024**3,
                        )
                        running_loss = 0.0
                        running_hit = 0
                        groups_since_log = 0

                    if (
                        config.save_every_steps > 0
                        and optimizer_step % config.save_every_steps == 0
                    ):
                        checkpoint_dir = (
                            output_dir / f"checkpoint-step-{optimizer_step}"
                        )
                        save_checkpoint(
                            model,
                            tokenizer,
                            checkpoint_dir,
                            optimizer_step,
                            epoch,
                        )

            val_metrics = evaluate(
                model,
                tokenizer,
                val_groups,
                config,
                yes_token_id,
                no_token_id,
                prefix_tokens,
                suffix_tokens,
                device,
            )
            if val_metrics:
                LOGGER.info(
                    "Validation epoch=%d/%d: loss=%.6f hit@1=%.4f mrr=%.4f",
                    epoch,
                    config.epochs,
                    val_metrics["loss"],
                    val_metrics["hit@1"],
                    val_metrics["mrr"],
                )
                selection_metric = val_metrics["hit@1"]
            else:
                selection_metric = -float(epoch)
                LOGGER.info("No validation split; saving epoch checkpoint.")

            if selection_metric > best_metric:
                best_metric = selection_metric
                best_metrics = val_metrics
                best_epoch = epoch
                best_optimizer_step = optimizer_step
                LOGGER.info(
                    "Updated best metric at epoch=%d step=%d: %.6f",
                    best_epoch,
                    best_optimizer_step,
                    best_metric,
                )

    final_metrics = evaluate(
        model,
        tokenizer,
        val_groups,
        config,
        yes_token_id,
        no_token_id,
        prefix_tokens,
        suffix_tokens,
        device,
    )
    save_checkpoint(
        model,
        tokenizer,
        final_dir,
        optimizer_step,
        config.epochs,
        {
            **(final_metrics or {}),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "best_optimizer_step": best_optimizer_step,
            "best_metrics": best_metrics or {},
        },
    )
    checkpoint_manifest.append(
        {
            "type": "final",
            "epoch": config.epochs,
            "optimizer_step": optimizer_step,
            "total_optimizer_steps": total_optimizer_steps,
            "path": str(final_dir),
            "metrics": final_metrics or {},
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "best_optimizer_step": best_optimizer_step,
            "best_metrics": best_metrics or {},
        }
    )
    write_checkpoint_manifest(checkpoint_manifest_path, checkpoint_manifest)

    duration = datetime.now() - start_time
    LOGGER.info("Training completed in %s", duration)
    LOGGER.info("Final model: %s", final_dir)


if __name__ == "__main__":
    main()
