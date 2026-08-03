# Skill-CARE

Skill-CARE is a SkillRouter-based skill routing package for large agent skill pools. It keeps the two-stage retrieve-and-rerank design: an embedding model first retrieves candidate skills from the full pool, then a reranker scores the top candidates with the full skill text.

This release includes evaluation code, benchmark metadata, and preparation assets for a reranker that has been further trained on a counterfactual dataset. The additional reranker training is intended to improve fine-grained discrimination between skills that look similar but differ in the capabilities required by a user request.

## What Is Included

- Two-stage retrieval and reranking evaluation code.
- Benchmark metadata and evaluation helpers.
- Data preparation files under `PREPARE_DATA/`.
- Training and evaluation scripts under `src/` and `scripts/`.

Model checkpoints are not committed to this repository. Put local model directories under `models/` when running experiments.

## Repository Layout

```text
.
|-- assets/              # README and paper assets
|-- data/                # Evaluation metadata and local benchmark shards
|-- evaluation/          # Evaluation protocol notes and examples
|-- manifests/           # Asset manifests
|-- PREPARE_DATA/        # Prepared reranker training data
|-- scripts/             # Data download and evaluation shell wrappers
|-- src/                 # Training, retrieval, reranking, and scoring code
|-- Makefile
|-- requirements.txt
`-- README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The evaluation scripts use PyTorch and run best on a CUDA-capable GPU.

## Model Paths

Place checkpoints under `models/`, for example:

```text
models/
|-- SkillRouter-Embedding-0.6B/
|-- SkillRouter-Reranker-0.6B/
|-- Counterfactual-SkillRouter-Reranker-0.6B-20260729-145634(use)/
`-- Counterfactual-SkillRouter-Reranker-0.6B-Final-v1-20260803-213247/
```

`SkillRouter-Embedding-0.6B` and `SkillRouter-Reranker-0.6B` are the original model directories before counterfactual reranker optimization. The two `Counterfactual-*` directories are example trained checkpoint directories used by the training and evaluation commands below.

The `models/` directory is intentionally ignored by git because model checkpoints are large. You can also point the scripts to any local checkpoint path with environment variables:

```bash
SKILLROUTER_EMB_MODEL_OR_PATH=/path/to/embedding \
SKILLROUTER_RERANK_MODEL_OR_PATH=/path/to/reranker \
bash scripts/evaluate_open_models.sh
```

## Counterfactual Reranker Training

The reranker is further trained on the counterfactual reranking dataset:

```text
PREPARE_DATA/counterfactual_rerank_training_pairs_latest.jsonl
```

For hosted or internal references, the same training-data file is:

```text
http://skill-care/PREPARE_DATA/counterfactual_rerank_training_pairs_latest.jsonl
```

Example training command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/root/.virtualenvs/SkillRouter/bin/python src/train_counterfactual_skillrouter_reranker.py \
  --train_file PREPARE_DATA/counterfactual_rerank_training_pairs_latest.jsonl \
  --model_path models/SkillRouter-Reranker-0.6B \
  --output_root models \
  --epochs 1 \
  --batch_size 16 \
  --gradient_accumulation_steps 4 \
  --max_length 4096
```

## Evaluation

Run the default retrieve-and-rerank pipeline:

```bash
bash scripts/evaluate_open_models.sh
```

or:

```bash
make eval-open-models
```

Evaluation on the original 75-task benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/root/.virtualenvs/SkillRouter/bin/python -m src.run_open_model_eval \
  --data_root data/eval_core \
  --encoder_model_or_path /root/autodl-tmp/SkillRouter/models/SkillRouter-Embedding-0.6B \
  --reranker_model_or_path /root/autodl-tmp/SkillRouter/models/Counterfactual-SkillRouter-Reranker-0.6B-Final-v1-20260803-213247/Counterfactual-SkillRouter-Reranker-0.6B-Final-v1-20260803-213247-final \
  --tiers easy hard \
  --retrieval_top_k 20 \
  --encoder_batch_size 32 \
  --reranker_batch_size 8
```

Evaluation on the newly collected Skill-CARE evaluation set:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/root/.virtualenvs/SkillRouter/bin/python -m src.run_open_model_eval \
  --data_root data/eval_new \
  --encoder_model_or_path /root/autodl-tmp/SkillRouter/models/SkillRouter-Embedding-0.6B \
  --reranker_model_or_path /root/autodl-tmp/SkillRouter/models/Counterfactual-SkillRouter-Reranker-0.6B-Final-v1-20260803-213247/Counterfactual-SkillRouter-Reranker-0.6B-Final-v1-20260803-213247-final \
  --tiers easy hard \
  --retrieval_top_k 20 \
  --encoder_batch_size 32 \
  --reranker_batch_size 8
```

Outputs are written under `outputs/` or the selected output directory. Runtime logs and experiment summaries are written under `exper_logs/` when enabled. These runtime artifacts are ignored by git.

## Data

The prepared counterfactual reranker training set is stored at:

```text
PREPARE_DATA/counterfactual_rerank_training_pairs_latest.jsonl
```

For hosted or internal references, the training-data directory is:

```text
http://skill-care/PREPARE_DATA/
```

and the training-data filename is:

```text
counterfactual_rerank_training_pairs_latest.jsonl
```

Large data files are tracked with Git LFS when needed.

Benchmark skill-pool shards can be downloaded with:

```bash
bash scripts/download_eval_data.sh
```

The evaluation metadata under `data/eval_core/` contains tasks, relevance labels, and manifest files used by the scoring code. The file `data/eval_new/tasks.jsonl` is our newly collected Skill-CARE evaluation dataset.

## Notes

- `models/` is local-only and must not be committed.
- `embeddings/` stores generated embedding caches and is local-only.
- `outputs/` and `exper_logs/` store runtime outputs and are local-only.
- The counterfactual reranker is described at a high level here; implementation details of data construction and training hyperparameters are intentionally not included in this README.

## License

See [LICENSE](LICENSE).
