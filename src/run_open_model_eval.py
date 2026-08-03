from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.common import (
    ensure_dir,
    format_query,
    format_rerank_prompt,
    format_skill,
    get_device,
    get_reranker_template_tokens,
    last_token_pool,
    load_embedding_model,
    load_reranker_model,
    tokenize_reranker_text,
)
from src.data_io import load_jsonl
from src.experiment_logs import RunLogger, create_run_dir, safe_path_part, write_json, write_jsonl
from src.metrics import compute_all_metrics


TIER_FILES = {
    "easy": "easy",
    "hard": "hard",
}
CACHE_VERSION = 1
SKILL_DESC_MAX = 500
SKILL_BODY_MAX = 8000
LOGGER: RunLogger | None = None


def aggregate(metrics_list: list[dict]) -> dict:
    out = {}
    if not metrics_list:
        return out
    for key in metrics_list[0]:
        out[key] = float(np.mean([m[key] for m in metrics_list]))
    out["count"] = len(metrics_list)
    return out


def log(message: str) -> None:
    if LOGGER is None:
        print(f"[SkillRouter Eval] {message}", flush=True)
    else:
        LOGGER.log(message)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def gpu_memory_status(device: torch.device) -> str:
    if device.type != "cuda":
        return "GPU memory: n/a on CPU"
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    return f"GPU memory: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB"


def hash_strings(values: list[str]) -> str:
    digest = hashlib.sha1()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def pool_embeddings_cache_path(
    cache_dir: Path,
    tier: str,
    encoder_model_or_path: str,
    encoder_max_length: int,
    pool_ids: list[str],
) -> Path:
    encoder_label = safe_path_part(Path(encoder_model_or_path).name or encoder_model_or_path)
    pool_hash = hash_strings(pool_ids)
    filename = (
        f"pool_{tier}_{encoder_label}_max{encoder_max_length}"
        f"_desc{SKILL_DESC_MAX}_body{SKILL_BODY_MAX}"
        f"_n{len(pool_ids)}_{pool_hash}.pt"
    )
    return cache_dir / filename


def load_cached_pool_embeddings(cache_path: Path, expected_pool_ids: list[str]) -> torch.Tensor | None:
    if not cache_path.exists():
        return None
    try:
        cache = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        log(f"Embedding cache load failed for {cache_path}: {exc}")
        return None

    pool_ids = cache.get("pool_ids")
    pool_embs = cache.get("pool_embs")
    if pool_ids != expected_pool_ids:
        log(f"Embedding cache ignored because pool ids differ: {cache_path}")
        return None
    if not isinstance(pool_embs, torch.Tensor):
        log(f"Embedding cache ignored because pool_embs is missing or invalid: {cache_path}")
        return None
    if pool_embs.shape[0] != len(expected_pool_ids):
        log(
            "Embedding cache ignored because row count differs: "
            f"{tuple(pool_embs.shape)} vs {len(expected_pool_ids)} ids"
        )
        return None
    return pool_embs


def save_pool_embeddings_cache(
    cache_path: Path,
    pool_embs: torch.Tensor,
    pool_ids: list[str],
    metadata: dict,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "pool_ids": pool_ids,
        "pool_embs": pool_embs.detach().cpu(),
        "metadata": metadata,
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(cache_path)


def encode_texts_with_progress(
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    model.eval()
    indexed_texts = sorted(enumerate(texts), key=lambda item: len(item[1]))
    all_embs: list[torch.Tensor | None] = [None] * len(texts)
    total_batches = (len(texts) + batch_size - 1) // batch_size
    progress = tqdm(
        range(0, len(indexed_texts), batch_size),
        total=total_batches,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
    )
    for i in progress:
        batch = indexed_texts[i:i + batch_size]
        batch_indices = [idx for idx, _ in batch]
        batch_texts = [text for _, text in batch]
        progress.set_postfix(items=f"{i + len(batch_texts)}/{len(indexed_texts)}")
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            embs = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
        for original_idx, emb in zip(batch_indices, embs.cpu()):
            all_embs[original_idx] = emb
    return torch.stack([emb for emb in all_embs if emb is not None], dim=0)


def metric_line(metrics: dict, group: str = "all") -> str:
    values = metrics.get(group, {})
    if not values:
        return f"{group}: n/a"
    parts = [
        f"Hit@1={values.get('Hit@1', 0.0):.3f}",
        f"MRR@10={values.get('MRR@10', 0.0):.3f}",
        f"R@10={values.get('Recall@10', 0.0):.3f}",
        f"count={int(values.get('count', 0))}",
    ]
    return f"{group}: " + ", ".join(parts)


def score_candidates_with_reranker(
    model,
    tokenizer,
    query_text: str,
    candidates: list[dict],
    prompt_format: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    prefix_tokens, suffix_tokens = get_reranker_template_tokens(tokenizer)
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")

    texts = [
        format_rerank_prompt(
            c["name"],
            c.get("description", c.get("desc", "")),
            c["body"],
            query_text,
            prompt_format=prompt_format,
        )
        for c in candidates
    ]
    tokenized = [
        tokenize_reranker_text(text, tokenizer, prefix_tokens, suffix_tokens, max_length)
        for text in texts
    ]

    scores: list[float] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for i in range(0, len(tokenized), batch_size):
        batch_ids = tokenized[i:i + batch_size]
        max_len = max(len(x) for x in batch_ids)
        padded, masks = [], []
        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded.append([pad_id] * pad_len + ids)
            masks.append([0] * pad_len + [1] * len(ids))
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
            batch_scores = (logits[:, token_true_id] - logits[:, token_false_id]).float().cpu().tolist()
        scores.extend(batch_scores)
    return scores


def main():
    global LOGGER

    parser = argparse.ArgumentParser(description="One-click local evaluation for the open SkillRouter 0.6B models.")
    parser.add_argument("--data_root", default="data/eval_core")
    parser.add_argument("--encoder_model_or_path", default="models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--reranker_model_or_path", default="models/SkillRouter-Reranker-0.6B")
    parser.add_argument("--task_mode", choices=["core", "all", "single"], default="core")
    parser.add_argument("--tiers", nargs="+", choices=["easy", "hard"], default=["easy", "hard"])
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--encoder_max_length", type=int, default=4096)
    parser.add_argument("--reranker_max_length", type=int, default=4096)
    parser.add_argument("--encoder_batch_size", type=int, default=32)
    parser.add_argument("--reranker_batch_size", type=int, default=8)
    parser.add_argument("--exper_logs_root", default="exper_logs")
    parser.add_argument("--embeddings_cache_dir", default="embeddings")
    parser.add_argument("--disable_embeddings_cache", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Legacy output root override. A timestamped running_* directory is still created inside it.",
    )
    parser.add_argument("--prompt_format", default="flat-full", choices=["flat-full", "flat-nd", "struct"])
    args = parser.parse_args()

    run_start = time.perf_counter()
    data_root = Path(args.data_root)
    embeddings_cache_dir = Path(args.embeddings_cache_dir)
    if not embeddings_cache_dir.is_absolute():
        embeddings_cache_dir = Path.cwd() / embeddings_cache_dir
    embeddings_cache_dir = ensure_dir(embeddings_cache_dir)
    encoder_label = Path(args.encoder_model_or_path).name or args.encoder_model_or_path
    reranker_label = Path(args.reranker_model_or_path).name or args.reranker_model_or_path
    output_dir = create_run_dir(
        "open-model-eval",
        [
            f"mode-{args.task_mode}",
            f"tiers-{'-'.join(args.tiers)}",
            f"top{args.retrieval_top_k}",
            f"emb-{encoder_label}",
            f"rank-{reranker_label}",
        ],
        args.output_dir or args.exper_logs_root,
    )
    LOGGER = RunLogger(output_dir, prefix="SkillRouter Eval")
    retrieval_dir = ensure_dir(output_dir / "retrieval")
    reranked_dir = ensure_dir(output_dir / "reranked")
    case_details_dir = ensure_dir(output_dir / "case_details")
    write_json(
        output_dir / "run_config.json",
        {
            "script": "src.run_open_model_eval",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_root": str(data_root.resolve()),
            "output_dir": str(output_dir.resolve()),
            "arguments": vars(args),
        },
    )

    device = get_device()
    log("Starting retrieve-and-rerank evaluation")
    log(f"Data root: {data_root.resolve()}")
    log(f"Output dir: {output_dir.resolve()}")
    log(f"Embeddings cache dir: {embeddings_cache_dir.resolve()}")
    log(f"Device: {device}")
    if device.type == "cuda":
        log(f"CUDA device: {torch.cuda.get_device_name(device)}")

    stage_start = time.perf_counter()
    log(f"Loading embedding model: {args.encoder_model_or_path}")
    try:
        emb_model, emb_tokenizer = load_embedding_model(args.encoder_model_or_path)
    except Exception as exc:
        raise SystemExit(
            "Failed to load the embedding model. "
            "Pass a valid local checkpoint path or a Hugging Face repo ID "
            "that contains model weights."
        ) from exc
    emb_model.to(device).eval()
    log(f"Embedding model ready in {format_duration(time.perf_counter() - stage_start)}; {gpu_memory_status(device)}")

    stage_start = time.perf_counter()
    log(f"Loading reranker model: {args.reranker_model_or_path}")
    try:
        rr_model, rr_tokenizer = load_reranker_model(args.reranker_model_or_path)
    except Exception as exc:
        raise SystemExit(
            "Failed to load the reranker model. "
            "Pass a valid local checkpoint path or a Hugging Face repo ID "
            "that contains model weights."
        ) from exc
    rr_model.to(device).eval()
    log(f"Reranker model ready in {format_duration(time.perf_counter() - stage_start)}; {gpu_memory_status(device)}")

    stage_start = time.perf_counter()
    log("Loading benchmark tasks and relevance labels")
    tasks = load_jsonl(data_root / "tasks.jsonl")
    relevance = json.loads((data_root / "relevance.json").read_text())
    log(f"Loaded {len(tasks)} tasks and {len(relevance)} relevance entries in {format_duration(time.perf_counter() - stage_start)}")

    filtered_tasks = []
    skipped_generic = 0
    skipped_non_single = 0
    for task in tqdm(tasks, desc="Filtering tasks", unit="task", dynamic_ncols=True):
        rel_entry = relevance.get(task["task_id"], {})
        task_type = rel_entry.get("task_type")
        if args.task_mode == "core" and task_type == "generic_only":
            skipped_generic += 1
            continue
        if args.task_mode == "single":
            gt_ids = set(rel_entry.get("gt_skill_ids", []))
            if len(gt_ids) != 1:
                skipped_non_single += 1
                continue
        filtered_tasks.append(task)
    log(
        "Task filter complete: "
        f"kept={len(filtered_tasks)}, skipped_generic={skipped_generic}, "
        f"skipped_non_single={skipped_non_single}, mode={args.task_mode}"
    )
    testset_summary = {
        "task_mode": args.task_mode,
        "total_tasks_in_tasks_jsonl": len(tasks),
        "relevance_entries": len(relevance),
        "filtered_tasks": len(filtered_tasks),
        "skipped_generic_only": skipped_generic,
        "skipped_non_single": skipped_non_single,
        "tiers": {},
    }

    query_texts = [format_query(t["instruction_text"], max_len=2000) for t in filtered_tasks]
    log(f"Encoding {len(query_texts)} query texts")
    query_embs = encode_texts_with_progress(
        emb_model,
        emb_tokenizer,
        query_texts,
        args.encoder_max_length,
        args.encoder_batch_size,
        device,
        desc="Encoding queries",
    )
    log(f"Query embeddings shape: {tuple(query_embs.shape)}")

    summary = {}
    for tier in tqdm(args.tiers, desc="Benchmark tiers", unit="tier", dynamic_ncols=True):
        tier_start = time.perf_counter()
        tier_stem = TIER_FILES[tier]
        log(f"[{tier}] Loading skill pool from {data_root / tier_stem}")
        pool = load_jsonl(data_root / tier_stem)
        pool_ids = [x["skill_id"] for x in pool]
        pool_id_set = set(pool_ids)
        log(f"[{tier}] Loaded {len(pool):,} candidate skills")
        cache_path = pool_embeddings_cache_path(
            embeddings_cache_dir,
            tier,
            args.encoder_model_or_path,
            args.encoder_max_length,
            pool_ids,
        )
        pool_embs = None
        if args.disable_embeddings_cache:
            log(f"[{tier}] Embedding cache disabled")
        else:
            log(f"[{tier}] Looking for embedding cache: {cache_path}")
            pool_embs = load_cached_pool_embeddings(cache_path, pool_ids)

        if pool_embs is None:
            pool_texts = [
                format_skill(x, desc_max=SKILL_DESC_MAX, body_max=SKILL_BODY_MAX)
                for x in tqdm(pool, desc=f"[{tier}] Formatting skills", unit="skill", dynamic_ncols=True)
            ]
            log(f"[{tier}] Encoding candidate skill pool")
            pool_embs = encode_texts_with_progress(
                emb_model,
                emb_tokenizer,
                pool_texts,
                args.encoder_max_length,
                args.encoder_batch_size,
                device,
                desc=f"[{tier}] Encoding skills",
            )
            if not args.disable_embeddings_cache:
                log(f"[{tier}] Saving embedding cache: {cache_path}")
                save_pool_embeddings_cache(
                    cache_path,
                    pool_embs,
                    pool_ids,
                    {
                        "tier": tier,
                        "encoder_model_or_path": args.encoder_model_or_path,
                        "encoder_max_length": args.encoder_max_length,
                        "desc_max": SKILL_DESC_MAX,
                        "body_max": SKILL_BODY_MAX,
                        "pool_size": len(pool_ids),
                        "pool_ids_hash": hash_strings(pool_ids),
                        "embedding_shape": list(pool_embs.shape),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
        else:
            log(f"[{tier}] Loaded pool embeddings from cache")
        log(f"[{tier}] Pool embeddings shape: {tuple(pool_embs.shape)}")
        log(f"[{tier}] Computing query x pool similarity matrix")
        sim_matrix = query_embs @ pool_embs.T
        log(f"[{tier}] Similarity matrix shape: {tuple(sim_matrix.shape)}")

        retrieval_results = {}
        reranked_results = {}
        metrics_retrieval = {"all": [], "single": [], "multi": []}
        metrics_pipeline = {"all": [], "single": [], "multi": []}
        skipped_no_gt = 0
        case_records = []

        task_progress = tqdm(
            enumerate(filtered_tasks),
            total=len(filtered_tasks),
            desc=f"[{tier}] Retrieve + rerank",
            unit="task",
            dynamic_ncols=True,
        )
        for qi, task in task_progress:
            task_id = task["task_id"]
            rel_entry = relevance[task_id]
            if args.task_mode == "core":
                raw_gt_ids = set(rel_entry.get("core_gt_ids", rel_entry.get("gt_skill_ids", [])))
            else:
                raw_gt_ids = set(rel_entry.get("gt_skill_ids", []))
            gt_ids = raw_gt_ids & pool_id_set
            if not gt_ids:
                skipped_no_gt += 1
                case_records.append(
                    {
                        "tier": tier,
                        "task_id": task_id,
                        "domain": task.get("domain"),
                        "instruction_text": task.get("instruction_text"),
                        "task_type": rel_entry.get("task_type"),
                        "difficulty": task.get("difficulty"),
                        "declared_num_skills": task.get("num_skills"),
                        "declared_skill_names": task.get("skill_names"),
                        "status": "skipped_no_ground_truth_in_pool",
                        "ground_truth_skill_ids": sorted(raw_gt_ids),
                        "ground_truth_skill_ids_in_pool": [],
                        "prediction_top1": None,
                        "retrieval_top_ids": [],
                        "reranked_top_ids": [],
                        "correct": False,
                    }
                )
                continue

            tier_relevance = {k: float(v) for k, v in rel_entry.get("relevance", {}).items() if k in pool_id_set}
            sims = sim_matrix[qi]
            _, topk_idx = torch.topk(sims, min(args.retrieval_top_k, len(pool_ids)))
            top_ids = [pool_ids[idx] for idx in topk_idx.tolist()]
            retrieval_results[task_id] = top_ids
            m_ret = compute_all_metrics(top_ids, gt_ids, tier_relevance or None)
            metrics_retrieval["all"].append(m_ret)
            if len(gt_ids) == 1:
                metrics_retrieval["single"].append(m_ret)
            else:
                metrics_retrieval["multi"].append(m_ret)

            candidates = [pool[idx] for idx in topk_idx.tolist()]
            scores = score_candidates_with_reranker(
                rr_model,
                rr_tokenizer,
                task["instruction_text"],
                candidates,
                args.prompt_format,
                args.reranker_max_length,
                args.reranker_batch_size,
                device,
            )
            ranked_pairs = sorted(zip(top_ids, scores), key=lambda x: x[1], reverse=True)
            reranked_ids = [rid for rid, _ in ranked_pairs]
            reranked_results[task_id] = reranked_ids
            m_pipe = compute_all_metrics(reranked_ids, gt_ids, tier_relevance or None)
            metrics_pipeline["all"].append(m_pipe)
            if len(gt_ids) == 1:
                metrics_pipeline["single"].append(m_pipe)
            else:
                metrics_pipeline["multi"].append(m_pipe)
            task_progress.set_postfix(
                valid=len(metrics_pipeline["all"]),
                hit1=f"{m_pipe['Hit@1']:.0f}",
                mrr=f"{m_pipe['MRR@10']:.2f}",
            )
            case_records.append(
                {
                    "tier": tier,
                    "task_id": task_id,
                    "domain": task.get("domain"),
                    "instruction_text": task.get("instruction_text"),
                    "task_type": rel_entry.get("task_type"),
                    "difficulty": task.get("difficulty"),
                    "declared_num_skills": task.get("num_skills"),
                    "declared_skill_names": task.get("skill_names"),
                    "status": "scored",
                    "ground_truth_skill_ids": sorted(raw_gt_ids),
                    "ground_truth_skill_ids_in_pool": sorted(gt_ids),
                    "relevance_labels_in_pool": tier_relevance,
                    "prediction_top1": reranked_ids[0] if reranked_ids else None,
                    "correct": bool(reranked_ids and reranked_ids[0] in gt_ids),
                    "retrieval_prediction_top1": top_ids[0] if top_ids else None,
                    "retrieval_correct": bool(top_ids and top_ids[0] in gt_ids),
                    "retrieval_top_ids": top_ids,
                    "reranked_top_ids": reranked_ids,
                    "reranker_scores": [
                        {"skill_id": rid, "score": float(score)}
                        for rid, score in ranked_pairs
                    ],
                    "retrieval_metrics": m_ret,
                    "pipeline_metrics": m_pipe,
                }
            )

        retrieval_path = retrieval_dir / f"{tier}.json"
        reranked_path = reranked_dir / f"{tier}.json"
        case_details_path = case_details_dir / f"{tier}_cases.jsonl"
        write_json(retrieval_path, retrieval_results)
        write_json(reranked_path, reranked_results)
        write_jsonl(case_details_path, case_records)
        summary[tier] = {
            "retrieval": {k: aggregate(v) for k, v in metrics_retrieval.items() if v},
            "pipeline": {k: aggregate(v) for k, v in metrics_pipeline.items() if v},
        }
        testset_summary["tiers"][tier] = {
            "pool_size": len(pool),
            "filtered_tasks": len(filtered_tasks),
            "scored_cases": len(metrics_pipeline["all"]),
            "single_cases": len(metrics_pipeline["single"]),
            "multi_cases": len(metrics_pipeline["multi"]),
            "skipped_no_ground_truth_in_pool": skipped_no_gt,
            "retrieval_predictions_file": str(retrieval_path),
            "reranked_predictions_file": str(reranked_path),
            "case_details_file": str(case_details_path),
        }
        log(f"[{tier}] Retrieval metrics: {metric_line(summary[tier]['retrieval'])}")
        log(f"[{tier}] Pipeline metrics: {metric_line(summary[tier]['pipeline'])}")
        log(
            f"[{tier}] Saved predictions to {retrieval_path} and {reranked_path}; "
            f"case details to {case_details_path}"
        )
        log(
            f"[{tier}] Finished in {format_duration(time.perf_counter() - tier_start)}; "
            f"scored={len(metrics_pipeline['all'])}, skipped_no_gt={skipped_no_gt}; "
            f"{gpu_memory_status(device)}"
        )

    summary_path = output_dir / "summary.json"
    testset_summary_path = output_dir / "testset_summary.json"
    write_json(summary_path, summary)
    write_json(testset_summary_path, testset_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {summary_path}")
    print(f"[saved] {testset_summary_path}")
    log(f"All done in {format_duration(time.perf_counter() - run_start)}")


if __name__ == "__main__":
    main()
