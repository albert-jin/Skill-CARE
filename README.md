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
`-- SkillRouter-Reranker-0.6B/
```

The `models/` directory is intentionally ignored by git because model checkpoints are large. You can also point the scripts to any local checkpoint path with environment variables:

```bash
SKILLROUTER_EMB_MODEL_OR_PATH=/path/to/embedding \
SKILLROUTER_RERANK_MODEL_OR_PATH=/path/to/reranker \
bash scripts/evaluate_open_models.sh
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

For a custom encoder and reranker:

```bash
python -m src.run_open_model_eval \
  --data_root data/eval_core \
  --encoder_model_or_path models/SkillRouter-Embedding-0.6B \
  --reranker_model_or_path models/SkillRouter-Reranker-0.6B \
  --tiers easy hard \
  --output_dir outputs/custom_pipeline_eval
```

Outputs are written under `outputs/` or the selected output directory. Runtime logs and experiment summaries are written under `exper_logs/` when enabled. These runtime artifacts are ignored by git.

## Data

The prepared reranker training file is stored under `PREPARE_DATA/`. Large data files are tracked with Git LFS when needed.

Benchmark skill-pool shards can be downloaded with:

```bash
bash scripts/download_eval_data.sh
```

The evaluation metadata under `data/eval_core/` contains tasks, relevance labels, and manifest files used by the scoring code.

## Notes

- `models/` is local-only and must not be committed.
- `embeddings/` stores generated embedding caches and is local-only.
- `outputs/` and `exper_logs/` store runtime outputs and are local-only.
- The counterfactual reranker is described at a high level here; implementation details of data construction and training hyperparameters are intentionally not included in this README.

## License

See [LICENSE](LICENSE).
