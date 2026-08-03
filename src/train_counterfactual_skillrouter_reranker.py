#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid final-v1 fine-tuning for SkillRouter-Reranker-0.6B.

Expected flat training fields:
    instruction_text
    skill_name
    skill_description
    skill_body
    label

Stage 1 uses listwise CE over the whole training file, grouped by instruction_text.
Stage 2 continues with pointwise CE over the appended tail rows only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (  # noqa: E402
    format_rerank_prompt,
    get_reranker_template_tokens,
    tokenize_reranker_text,
)
from src import train_counterfactual_skillrouter_reranker as listwise_train  # noqa: E402


LOGGER = logging.getLogger("train_skillrouter_reranker_final_v1")
DEFAULT_TRAIN_FILE = (
    "PAPER_PREPARE/DATA_AUGMENT/counterfactual_rerank_training_pairs_latest.jsonl"
)
TRAINING_FIELDS = [
    "instruction_text",
    "skill_name",
    "skill_description",
    "skill_body",
    "label",
]
REQUIRED_TEXT_FIELDS = TRAINING_FIELDS[:-1]


@dataclass
class PointwiseExample:
    instruction_text: str
    skill_name: str
    skill_description: str
    skill_body: str
    label: int


@dataclass
class TrainConfig:
    model_path: str
    train_file: str
    output_dir: str
    run_name_prefix: str
    max_length: int
    description_max_chars: int
    body_max_chars: int
    group_size: int
    candidate_micro_batch_size: int
    gradient_accumulation_groups: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    second_stage_epochs: int
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
    fail_on_empty_text: bool
    filter_false_negatives: bool
    body_jaccard_threshold: float
    log_every_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listwise then pointwise-tail fine-tuning for SkillRouter-Reranker-0.6B"
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
        default=DEFAULT_TRAIN_FILE,
        help="Training data: jsonl/json/csv/parquet.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="models",
        help="The timestamped run directory is created below this directory.",
    )
    parser.add_argument(
        "--run_name_prefix",
        type=str,
        default="Counterfactual-SkillRouter-Reranker-0.6B-Final-v1",
    )

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--description_max_chars", type=int, default=500)
    parser.add_argument("--body_max_chars", type=int, default=2000)
    parser.add_argument("--group_size", type=int, default=20)
    parser.add_argument("--candidate_micro_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_groups", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--second_stage_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--listwise_temperature", type=float, default=1.0)
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.0,
        help=(
            "Validation ratio for listwise groups. Default 0 trains all groups."
        ),
    )
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
        "--fail_on_empty_text",
        action="store_true",
        help="Fail instead of skipping rows with empty instruction/skill fields.",
    )
    parser.add_argument(
        "--filter_false_negatives",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--body_jaccard_threshold", type=float, default=0.6)
    parser.add_argument("--log_every_steps", type=int, default=10)
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


def build_examples(rows: Sequence[Dict[str, Any]], fail_on_empty_text: bool) -> List[PointwiseExample]:
    if not rows:
        raise ValueError("Training file is empty.")

    missing = set(TRAINING_FIELDS) - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    examples: List[PointwiseExample] = []
    label_counts = {0: 0, 1: 0}
    skipped = 0
    for row_idx, row in enumerate(rows, start=1):
        empty_fields = [
            field
            for field in REQUIRED_TEXT_FIELDS
            if not normalize_text(row.get(field))
        ]
        if empty_fields:
            message = f"row {row_idx} has empty required fields: {empty_fields}"
            if fail_on_empty_text:
                raise ValueError(message)
            skipped += 1
            LOGGER.warning("Skip %s", message)
            continue

        label = parse_label(row.get("label"))
        label_counts[label] += 1
        examples.append(
            PointwiseExample(
                instruction_text=normalize_text(row.get("instruction_text")),
                skill_name=normalize_text(row.get("skill_name")),
                skill_description=normalize_text(row.get("skill_description")),
                skill_body=normalize_text(row.get("skill_body")),
                label=label,
            )
        )

    LOGGER.info(
        "Built %d ordered pointwise examples from %d rows; skipped_empty=%d, label0=%d, label1=%d",
        len(examples),
        len(rows),
        skipped,
        label_counts[0],
        label_counts[1],
    )
    if not examples:
        raise ValueError("No valid pointwise examples were created.")
    if label_counts[0] == 0 or label_counts[1] == 0:
        LOGGER.warning(
            "Training data has label0=%d, label1=%d; pointwise CE works best with both classes.",
            label_counts[0],
            label_counts[1],
        )
    return examples


class PointwiseDataset(Dataset):
    def __init__(self, examples: Sequence[PointwiseExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PointwiseExample:
        return self.examples[index]


def identity_collate(items: List[PointwiseExample]) -> List[PointwiseExample]:
    return items


def split_examples_in_order(
    examples: List[PointwiseExample], val_ratio: float
) -> Tuple[List[PointwiseExample], List[PointwiseExample]]:
    if val_ratio <= 0 or len(examples) < 2:
        return examples, []
    if val_ratio >= 1:
        raise ValueError("--val_ratio must be less than 1.")

    val_size = max(1, int(round(len(examples) * val_ratio)))
    val_size = min(val_size, len(examples) - 1)
    return examples[val_size:], examples[:val_size]


def build_reranker_prompt(
    example: PointwiseExample,
    description_max_chars: int,
    body_max_chars: int,
) -> str:
    return format_rerank_prompt(
        example.skill_name,
        example.skill_description,
        example.skill_body,
        example.instruction_text,
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
    padded: List[List[int]] = []
    attention_masks: List[List[int]] = []
    for ids in encoded_ids:
        pad_len = batch_max_len - len(ids)
        padded.append([pad_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))
    return {
        "input_ids": torch.tensor(padded, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
    }


def get_base_transformer(model: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise AttributeError(
            "Expected a causal LM with `.model` and `.lm_head` attributes."
        )
    return model.model


def answer_logits_for_prompts(
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
    answer_token_ids = torch.tensor(
        [no_token_id, yes_token_id],
        dtype=torch.long,
        device=model.lm_head.weight.device,
    )
    answer_weight = model.lm_head.weight.index_select(0, answer_token_ids)
    answer_bias = getattr(model.lm_head, "bias", None)
    if answer_bias is not None:
        answer_bias = answer_bias.index_select(0, answer_token_ids)
    return F.linear(last_hidden, answer_weight, answer_bias)


def batch_prompts_and_labels(
    batch: Sequence[PointwiseExample],
    config: TrainConfig,
    device: torch.device,
) -> Tuple[List[str], torch.Tensor]:
    prompts = [
        build_reranker_prompt(
            example,
            config.description_max_chars,
            config.body_max_chars,
        )
        for example in batch
    ]
    labels = torch.tensor(
        [example.label for example in batch],
        dtype=torch.long,
        device=device,
    )
    return prompts, labels


def train_one_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: Sequence[PointwiseExample],
    config: TrainConfig,
    yes_token_id: int,
    no_token_id: int,
    prefix_tokens: Sequence[int],
    suffix_tokens: Sequence[int],
    device: torch.device,
    loss_scale: float,
) -> Tuple[float, float]:
    prompts, labels = batch_prompts_and_labels(batch, config, device)
    logits = answer_logits_for_prompts(
        model,
        tokenizer,
        prompts,
        config.max_length,
        yes_token_id,
        no_token_id,
        prefix_tokens,
        suffix_tokens,
        device,
    )
    loss = F.cross_entropy(logits.float(), labels)
    (loss * loss_scale).backward()

    predictions = torch.argmax(logits.detach(), dim=1)
    accuracy = (predictions == labels).float().mean().item()
    return float(loss.item()), float(accuracy)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: Sequence[PointwiseExample],
    config: TrainConfig,
    yes_token_id: int,
    no_token_id: int,
    prefix_tokens: Sequence[int],
    suffix_tokens: Sequence[int],
    device: torch.device,
) -> Dict[str, float]:
    if not examples:
        return {}

    model.eval()
    loader = DataLoader(
        PointwiseDataset(examples),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=identity_collate,
        pin_memory=True,
    )

    total_loss = 0.0
    total_count = 0
    total_correct = 0
    class_total = {0: 0, 1: 0}
    class_correct = {0: 0, 1: 0}
    yes_prob_sum = 0.0

    for batch in loader:
        prompts, labels = batch_prompts_and_labels(batch, config, device)
        logits = answer_logits_for_prompts(
            model,
            tokenizer,
            prompts,
            config.max_length,
            yes_token_id,
            no_token_id,
            prefix_tokens,
            suffix_tokens,
            device,
        )
        loss = F.cross_entropy(logits.float(), labels, reduction="sum")
        probs = torch.softmax(logits.float(), dim=1)
        predictions = torch.argmax(logits, dim=1)
        correct = predictions == labels

        total_loss += float(loss.item())
        total_count += int(labels.numel())
        total_correct += int(correct.sum().item())
        yes_prob_sum += float(probs[:, 1].sum().item())
        for label_value in (0, 1):
            mask = labels == label_value
            class_total[label_value] += int(mask.sum().item())
            class_correct[label_value] += int((correct & mask).sum().item())

    return {
        "loss": total_loss / max(1, total_count),
        "accuracy": total_correct / max(1, total_count),
        "negative_accuracy": class_correct[0] / max(1, class_total[0]),
        "positive_accuracy": class_correct[1] / max(1, class_total[1]),
        "mean_yes_probability": yes_prob_sum / max(1, total_count),
        "count": float(total_count),
    }


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(output_dir)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def first_stage_lr(base_lr: float, optimizer_step: int, total_steps: int) -> float:
    progress = optimizer_step / max(1, total_steps)
    for boundary, multiplier in ((0.5, 0.1), (0.75, 0.3), (0.9, 0.5), (1.01, 1.0)):
        if progress < boundary:
            return base_lr * multiplier
    return base_lr


def infer_tail_focus_count(train_file: Path, train_count: int) -> int:
    if train_count <= 0:
        return 0

    summary_path = train_file.with_suffix(".summary.json")
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            append_counts = summary.get("append_counts", {})
            added = sum(
                int(value)
                for key, value in append_counts.items()
                if str(key).endswith("_rows_added")
            )
            if added > 0:
                return min(train_count, added)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not read tail-count summary %s: %s", summary_path, exc)

    return max(1, math.ceil(train_count * 0.1))


def load_model(model_path: Path, config: TrainConfig) -> torch.nn.Module:
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except (ImportError, ValueError) as exc:
        if not config.use_flash_attention_2:
            raise
        LOGGER.warning("FlashAttention-2 unavailable (%s). Falling back to SDPA.", exc)
        model_kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    return model


def main() -> None:
    args = parse_args()
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d-%H%M%S")
    output_root = Path(args.output_root)
    run_name = f"{args.run_name_prefix}-{timestamp}"
    output_dir = output_root / run_name
    final_dir = output_dir / f"{run_name}-final"
    setup_logging(output_dir)
    set_seed(args.seed)

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1.")
    if args.group_size < 1:
        raise ValueError("--group_size must be >= 1.")
    if args.candidate_micro_batch_size < 1:
        raise ValueError("--candidate_micro_batch_size must be >= 1.")
    if args.gradient_accumulation_groups < 1:
        raise ValueError("--gradient_accumulation_groups must be >= 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.second_stage_epochs < 1:
        raise ValueError("--second_stage_epochs must be >= 1.")
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
        run_name_prefix=args.run_name_prefix,
        max_length=args.max_length,
        description_max_chars=args.description_max_chars,
        body_max_chars=args.body_max_chars,
        group_size=args.group_size,
        candidate_micro_batch_size=args.candidate_micro_batch_size,
        gradient_accumulation_groups=args.gradient_accumulation_groups,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        second_stage_epochs=args.second_stage_epochs,
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
        fail_on_empty_text=args.fail_on_empty_text,
        filter_false_negatives=args.filter_false_negatives,
        body_jaccard_threshold=args.body_jaccard_threshold,
        log_every_steps=args.log_every_steps,
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
    LOGGER.info("Answer token ids: yes=%d, no=%d", yes_token_id, no_token_id)
    LOGGER.info(
        "Reranker template token lengths: prefix=%d, suffix=%d",
        len(prefix_tokens),
        len(suffix_tokens),
    )

    LOGGER.info("Loading model from %s", model_path)
    model = load_model(model_path, config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(
        "Model params: total=%s, trainable=%s",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    rows = load_rows(train_file)
    examples = build_examples(rows, config.fail_on_empty_text)
    tail_focus_count = infer_tail_focus_count(train_file, len(examples))
    tail_focus_count = min(tail_focus_count, len(examples))
    tail_start_index = max(0, len(examples) - tail_focus_count)
    tail_examples = examples[tail_start_index:]

    groups = listwise_train.build_groups(
        rows,
        group_size=config.group_size,
        filter_false_negatives=config.filter_false_negatives,
        body_jaccard_threshold=config.body_jaccard_threshold,
    )
    train_groups, _ = listwise_train.split_groups(groups, config.val_ratio, config.seed)
    listwise_loader = DataLoader(
        listwise_train.GroupDataset(train_groups),
        batch_size=1,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=listwise_train.identity_collate,
        pin_memory=True,
    )
    tail_loader = DataLoader(
        PointwiseDataset(tail_examples),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=identity_collate,
        pin_memory=True,
    )

    listwise_optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate * 0.1,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    listwise_steps_per_epoch = math.ceil(
        len(listwise_loader) / config.gradient_accumulation_groups
    )
    listwise_total_steps = listwise_steps_per_epoch * config.epochs

    pointwise_optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    pointwise_steps_per_epoch = math.ceil(
        len(tail_loader) / config.gradient_accumulation_steps
    )
    pointwise_total_steps = pointwise_steps_per_epoch * config.second_stage_epochs

    total_progress_units = (
        len(listwise_loader) * config.epochs
        + len(tail_loader) * config.second_stage_epochs
    )
    total_optimizer_steps = listwise_total_steps + pointwise_total_steps
    LOGGER.info(
        "Training: epochs=%d, groups/epoch=%d, total_units=%d, "
        "optimizer_steps=%d, max_length=%d",
        config.epochs,
        len(listwise_loader),
        total_progress_units,
        total_optimizer_steps,
        config.max_length,
    )

    listwise_optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0

    with tqdm(
        total=total_progress_units,
        desc=f"Epoch 1/{config.epochs}",
        unit="unit",
        dynamic_ncols=True,
    ) as train_progress:
        for epoch in range(1, config.epochs + 1):
            model.train()
            running_loss = 0.0
            running_hit = 0.0
            groups_since_log = 0

            train_progress.set_description(f"Epoch {epoch}/{config.epochs}")
            for group_idx, group in enumerate(listwise_loader, start=1):
                set_optimizer_lr(
                    listwise_optimizer,
                    first_stage_lr(
                        config.learning_rate,
                        optimizer_step,
                        listwise_total_steps,
                    ),
                )
                loss_scale = 1.0 / config.gradient_accumulation_groups
                loss_value, hit1 = listwise_train.train_one_group_exact_two_pass(
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
                running_hit += float(hit1)
                groups_since_log += 1

                avg_loss = running_loss / max(1, groups_since_log)
                avg_hit = running_hit / max(1, groups_since_log)
                train_progress.set_postfix(
                    epoch=f"{epoch}/{config.epochs}",
                    unit=f"{train_progress.n}/{total_progress_units}",
                    opt_step=f"{optimizer_step}/{total_optimizer_steps}",
                    loss=f"{avg_loss:.4f}",
                    hit1=f"{avg_hit:.3f}",
                    refresh=False,
                )
                train_progress.update(1)

                should_step = (
                    group_idx % config.gradient_accumulation_groups == 0
                    or group_idx == len(listwise_loader)
                )
                if should_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                    listwise_optimizer.step()
                    listwise_optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                    train_progress.set_postfix(
                        epoch=f"{epoch}/{config.epochs}",
                        unit=f"{train_progress.n}/{total_progress_units}",
                        opt_step=f"{optimizer_step}/{total_optimizer_steps}",
                        loss=f"{avg_loss:.4f}",
                        hit1=f"{avg_hit:.3f}",
                        grad_norm=f"{float(grad_norm):.2f}",
                        refresh=False,
                    )

                    if (
                        config.log_every_steps > 0
                        and optimizer_step % config.log_every_steps == 0
                    ):
                        LOGGER.info(
                            "epoch=%d/%d step=%d/%d group=%d/%d "
                            "loss=%.6f hit@1=%.4f grad_norm=%.4f "
                            "gpu_alloc=%.2fGB gpu_reserved=%.2fGB",
                            epoch,
                            config.epochs,
                            optimizer_step,
                            total_optimizer_steps,
                            group_idx,
                            len(listwise_loader),
                            avg_loss,
                            avg_hit,
                            float(grad_norm),
                            torch.cuda.memory_allocated() / 1024**3,
                            torch.cuda.memory_reserved() / 1024**3,
                        )
                        running_loss = 0.0
                        running_hit = 0.0
                        groups_since_log = 0

        pointwise_optimizer.zero_grad(set_to_none=True)
        train_progress.set_postfix_str("", refresh=False)
        for _ in range(config.second_stage_epochs):
            model.train()
            accumulation_batches = 0
            for batch_idx, batch in enumerate(tail_loader, start=1):
                accumulation_batches += 1
                loss_scale = 1.0 / config.gradient_accumulation_steps
                train_one_batch(
                    model,
                    tokenizer,
                    batch,
                    config,
                    yes_token_id,
                    no_token_id,
                    prefix_tokens,
                    suffix_tokens,
                    device,
                    loss_scale=loss_scale,
                )
                train_progress.update(1)

                should_step = (
                    accumulation_batches >= config.gradient_accumulation_steps
                    or batch_idx == len(tail_loader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                    pointwise_optimizer.step()
                    pointwise_optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    accumulation_batches = 0

    save_checkpoint(model, tokenizer, final_dir)
    LOGGER.info("Final model: %s", final_dir)


if __name__ == "__main__":
    main()
