from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.data_io import load_jsonl
from src.experiment_logs import RunLogger, create_run_dir, write_json, write_jsonl
from src.metrics import compute_all_metrics


TIER_FILES = {
    "easy": "easy",
    "hard": "hard",
}


def aggregate(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}
    out = {}
    for key in metrics_list[0]:
        out[key] = float(np.mean([m[key] for m in metrics_list]))
    out["count"] = len(metrics_list)
    return out


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval or reranked predictions on SkillRouter Eval Core.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--task_mode", choices=["core", "all", "single"], default="core")
    parser.add_argument("--tier", choices=["easy", "hard"], required=True)
    parser.add_argument("--exper_logs_root", default="exper_logs")
    parser.add_argument("--output_json", default="", help="Optional extra summary JSON path. A run folder is always written under exper_logs.")
    args = parser.parse_args()

    run_start = time.perf_counter()
    data_root = Path(args.data_root)
    predictions_path = Path(args.predictions)
    run_dir = create_run_dir(
        "score-predictions",
        [
            f"tier-{args.tier}",
            f"mode-{args.task_mode}",
            f"pred-{predictions_path.stem}",
        ],
        args.exper_logs_root,
    )
    logger = RunLogger(run_dir, prefix="SkillRouter Score")
    logger.log("Starting prediction-file evaluation")
    logger.log(f"Data root: {data_root.resolve()}")
    logger.log(f"Predictions: {predictions_path.resolve()}")
    logger.log(f"Run dir: {run_dir.resolve()}")
    write_json(
        run_dir / "run_config.json",
        {
            "script": "src.evaluate_predictions",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_root": str(data_root.resolve()),
            "predictions": str(predictions_path.resolve()),
            "run_dir": str(run_dir.resolve()),
            "arguments": vars(args),
        },
    )

    logger.log(f"Loading tier pool: {args.tier}")
    tier_pool = load_jsonl(data_root / TIER_FILES[args.tier])
    pool_id_set = {x["skill_id"] for x in tier_pool}
    logger.log(f"Loaded {len(tier_pool):,} candidate skills")
    tasks = load_jsonl(data_root / "tasks.jsonl")
    relevance = json.loads((data_root / "relevance.json").read_text())
    predictions = json.loads(predictions_path.read_text())
    logger.log(f"Loaded {len(tasks)} tasks, {len(relevance)} relevance entries, {len(predictions)} prediction entries")

    results_by_stratum = {"all": [], "single": [], "multi": []}
    case_records = []
    skipped_generic = 0
    skipped_non_single = 0
    skipped_no_gt = 0
    missing_prediction = 0
    for task in tqdm(tasks, desc=f"Scoring {args.tier}", unit="task", dynamic_ncols=True):
        task_id = task["task_id"]
        rel_entry = relevance.get(task_id, {})
        task_type = rel_entry.get("task_type")
        if args.task_mode == "core":
            if task_type == "generic_only":
                skipped_generic += 1
                continue
            gt_ids = set(rel_entry.get("core_gt_ids", rel_entry.get("gt_skill_ids", [])))
        elif args.task_mode == "single":
            gt_ids = set(rel_entry.get("gt_skill_ids", []))
            if len(gt_ids) != 1:
                skipped_non_single += 1
                continue
        else:
            gt_ids = set(rel_entry.get("gt_skill_ids", []))

        gt_ids_in_pool = gt_ids & pool_id_set
        base_record = {
            "tier": args.tier,
            "task_id": task_id,
            "domain": task.get("domain"),
            "instruction_text": task.get("instruction_text"),
            "task_type": task_type,
            "difficulty": task.get("difficulty"),
            "declared_num_skills": task.get("num_skills"),
            "declared_skill_names": task.get("skill_names"),
            "ground_truth_skill_ids": sorted(gt_ids),
            "ground_truth_skill_ids_in_pool": sorted(gt_ids_in_pool),
        }
        if not gt_ids_in_pool:
            skipped_no_gt += 1
            case_records.append(
                {
                    **base_record,
                    "status": "skipped_no_ground_truth_in_pool",
                    "prediction_top1": None,
                    "ranked_ids": [],
                    "correct": False,
                }
            )
            continue
        if task_id not in predictions:
            missing_prediction += 1
            case_records.append(
                {
                    **base_record,
                    "status": "missing_prediction",
                    "prediction_top1": None,
                    "ranked_ids": [],
                    "correct": False,
                }
            )
            continue

        ranked_ids = predictions[task_id]
        tier_relevance = {
            k: float(v) for k, v in rel_entry.get("relevance", {}).items() if k in pool_id_set
        }
        metrics = compute_all_metrics(ranked_ids, gt_ids_in_pool, tier_relevance or None)
        correct = bool(ranked_ids and ranked_ids[0] in gt_ids_in_pool)
        results_by_stratum["all"].append(metrics)
        if len(gt_ids) == 1:
            results_by_stratum["single"].append(metrics)
        else:
            results_by_stratum["multi"].append(metrics)
        case_records.append(
            {
                **base_record,
                "status": "scored",
                "relevance_labels_in_pool": tier_relevance,
                "prediction_top1": ranked_ids[0] if ranked_ids else None,
                "ranked_ids": ranked_ids,
                "correct": correct,
                "metrics": metrics,
            }
        )

    aggregated = {k: aggregate(v) for k, v in results_by_stratum.items() if v}
    testset_summary = {
        "task_mode": args.task_mode,
        "tier": args.tier,
        "total_tasks_in_tasks_jsonl": len(tasks),
        "relevance_entries": len(relevance),
        "prediction_entries": len(predictions),
        "pool_size": len(tier_pool),
        "scored_cases": len(results_by_stratum["all"]),
        "single_cases": len(results_by_stratum["single"]),
        "multi_cases": len(results_by_stratum["multi"]),
        "skipped_generic_only": skipped_generic,
        "skipped_non_single": skipped_non_single,
        "skipped_no_ground_truth_in_pool": skipped_no_gt,
        "missing_prediction": missing_prediction,
    }
    summary_path = run_dir / "summary.json"
    testset_summary_path = run_dir / "testset_summary.json"
    case_details_path = run_dir / "case_details.jsonl"
    write_json(summary_path, aggregated)
    write_json(testset_summary_path, testset_summary)
    write_jsonl(case_details_path, case_records)

    text = json.dumps(aggregated, indent=2)
    print(text)
    logger.log(f"Metrics: {metric_line(aggregated)}")
    logger.log(f"Saved summary to {summary_path}")
    logger.log(f"Saved testset summary to {testset_summary_path}")
    logger.log(f"Saved case details to {case_details_path}")
    if args.output_json:
        write_json(args.output_json, aggregated)
        logger.log(f"Also wrote summary to --output_json path: {args.output_json}")
    logger.log(f"All done in {time.perf_counter() - run_start:.1f}s")


if __name__ == "__main__":
    main()
