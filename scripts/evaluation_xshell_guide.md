# SkillRouter 远程评测复现指南

这份说明面向在远程服务器、Xshell、tmux 或 screen 里直接运行 SkillRouter 评测。默认假设你已经进入仓库根目录：

```bash
cd /path/to/SkillRouter
```

## 结果存放在哪里

现在两个评测入口都会默认把结果写到仓库根目录的 `exper_logs/` 下面。每次运行都会创建一个新的 run 文件夹，名称类似：

```text
exper_logs/running_open-model-eval_mode-core_tiers-easy-hard_top20_emb-SkillRouter-Embedding-0.6B_rank-SkillRouter-Reranker-0.6B_20260725_153012/
exper_logs/running_score-predictions_tier-easy_mode-core_pred-easy_20260725_153245/
```

端到端模型评测 run 目录通常包含：

```text
runtime.log
run_config.json
summary.json
testset_summary.json
retrieval/easy.json
retrieval/hard.json
reranked/easy.json
reranked/hard.json
case_details/easy_cases.jsonl
case_details/hard_cases.jsonl
```

已有 prediction 文件打分 run 目录通常包含：

```text
runtime.log
run_config.json
summary.json
testset_summary.json
case_details.jsonl
```

其中：

- `runtime.log`：运行时记录，包括加载数据、任务数量、输出路径和最终指标。
- `summary.json`：聚合指标，例如 `Hit@1`、`MRR@10`、`Recall@10`。
- `testset_summary.json`：测试集统计，包括 pool size、scored cases、single/multi 数量、跳过数量等。
- `case_details/*.jsonl`：逐条测试样例详情，包括题目文本、domain、预测结果、真实 skill id、top-1 是否正确、完整 ranked ids 和该样例指标。

## `Path(args.output_json).write_text(text)` 原来默认在哪里

在旧版 `src/evaluate_predictions.py` 里，`--output_json` 默认是空字符串，所以默认不会写文件，只会把指标打印到终端。

如果你手动传：

```bash
--output_json outputs/metrics.json
```

那么 `Path(args.output_json).write_text(text)` 会按当前工作目录解析相对路径。因为 `scripts/evaluate_predictions.sh` 会先 `cd` 到仓库根目录，所以相对路径默认就是相对于仓库根目录。

现在代码已经改成：即使不传 `--output_json`，也会自动写入本次运行的 `exper_logs/running_.../summary.json`。`--output_json` 只作为额外再导出一份 summary 的兼容选项。

## 两个 shell 脚本的区别

### `scripts/evaluate_open_models.sh`

这是测试模型 checkpoint 的端到端脚本。它调用：

```bash
python3 -m src.run_open_model_eval
```

它会完整执行：

1. 加载 embedding checkpoint。
2. 加载 reranker checkpoint。
3. 读取 `tasks.jsonl` 和 `relevance.json`。
4. 在 `easy` / `hard` skill pool 上做向量检索。
5. 对 top-k candidates 做 rerank。
6. 写出预测文件、聚合指标、逐 case 详情和运行日志。

如果你要测试作者发布的模型，或者测试你自己训练出来的 encoder/reranker checkpoint，跑这个脚本。

### `scripts/evaluate_predictions.sh`

这是只给已有预测文件打分的脚本。它调用：

```bash
python3 -m src.evaluate_predictions
```

它不会加载模型，也不会做 embedding/retrieval/rerank。它只读取一个已经生成好的 prediction JSON，例如：

```json
{
  "task_id_1": ["skill_id_a", "skill_id_b", "skill_id_c"],
  "task_id_2": ["skill_id_x", "skill_id_y", "skill_id_z"]
}
```

然后根据 `relevance.json` 计算指标，并写出逐 case 对比文件。

结论：测试训练好的 checkpoint 用 `evaluate_open_models.sh`；只评估已有预测结果用 `evaluate_predictions.sh`。

## 1. 准备环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

如果你使用 conda：

```bash
conda create -n skillrouter python=3.10 -y
conda activate skillrouter
pip install -r requirements.txt
```

## 2. 下载评测数据

如果 `data/eval_core/easy/` 和 `data/eval_core/hard/` 下没有 `.jsonl.gz` 分片，先下载：

```bash
bash scripts/download_eval_data.sh
```

检查：

```bash
find data/eval_core/easy -name '*.jsonl.gz' | head
find data/eval_core/hard -name '*.jsonl.gz' | head
```

## 3. 跑作者发布的 0.6B 模型

如果本地已经有模型目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_open_models.sh
```

默认读取：

```text
models/SkillRouter-Embedding-0.6B
models/SkillRouter-Reranker-0.6B
```

如果想直接从 Hugging Face 模型 ID 读取：

```bash
CUDA_VISIBLE_DEVICES=0 \
SKILLROUTER_EMB_MODEL_OR_PATH=pipizhao/SkillRouter-Embedding-0.6B \
SKILLROUTER_RERANK_MODEL_OR_PATH=pipizhao/SkillRouter-Reranker-0.6B \
bash scripts/evaluate_open_models.sh
```

## 4. 测试你自己训练的 checkpoint

假设你的模型路径是：

```text
/mnt/checkpoints/my-skillrouter-emb
/mnt/checkpoints/my-skillrouter-rerank
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
SKILLROUTER_EMB_MODEL_OR_PATH=/mnt/checkpoints/my-skillrouter-emb \
SKILLROUTER_RERANK_MODEL_OR_PATH=/mnt/checkpoints/my-skillrouter-rerank \
bash scripts/evaluate_open_models.sh
```

如果显存紧张：

```bash
CUDA_VISIBLE_DEVICES=0 \
SKILLROUTER_EMB_MODEL_OR_PATH=/mnt/checkpoints/my-skillrouter-emb \
SKILLROUTER_RERANK_MODEL_OR_PATH=/mnt/checkpoints/my-skillrouter-rerank \
bash scripts/evaluate_open_models.sh \
  --encoder_batch_size 8 \
  --reranker_batch_size 2
```

如果只想快速跑一个 tier：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_open_models.sh --tiers easy
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_open_models.sh --tiers hard
```

## 5. 后台运行

推荐用 `tmux`：

```bash
tmux new -s skillrouter_eval
cd /path/to/SkillRouter
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_open_models.sh
```

退出但保留任务：

```bash
Ctrl-b d
```

重新进入：

```bash
tmux attach -t skillrouter_eval
```

也可以用 `nohup`：

```bash
nohup bash -lc 'source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_open_models.sh' \
  > exper_logs/nohup_open_model_eval.log 2>&1 &
```

查看：

```bash
tail -f exper_logs/nohup_open_model_eval.log
```

## 6. 只对已有 prediction JSON 打分

端到端脚本跑完后，会生成：

```text
exper_logs/running_open-model-eval_.../reranked/easy.json
exper_logs/running_open-model-eval_.../reranked/hard.json
```

给 easy 结果打分：

```bash
bash scripts/evaluate_predictions.sh \
  --predictions exper_logs/running_open-model-eval_具体时间/reranked/easy.json \
  --tier easy
```

给 hard 结果打分：

```bash
bash scripts/evaluate_predictions.sh \
  --predictions exper_logs/running_open-model-eval_具体时间/reranked/hard.json \
  --tier hard
```

如果你想额外导出一份 summary 到固定路径：

```bash
bash scripts/evaluate_predictions.sh \
  --predictions exper_logs/running_open-model-eval_具体时间/reranked/easy.json \
  --tier easy \
  --output_json exper_logs/manual_easy_metrics.json
```

## 7. 常用参数

`evaluate_open_models.sh` 会把额外参数转发给 `src.run_open_model_eval`。常用参数：

```bash
--tiers easy hard
--task_mode core
--retrieval_top_k 20
--encoder_batch_size 32
--reranker_batch_size 8
--encoder_max_length 4096
--reranker_max_length 4096
--prompt_format flat-full
--exper_logs_root exper_logs
```

显存不够时优先调小：

```bash
--encoder_batch_size 8 --reranker_batch_size 2
```

复现论文 core setting，保持：

```bash
--task_mode core --tiers easy hard --retrieval_top_k 20 --prompt_format flat-full
```

## 8. 最小复现命令合集

从零开始：

```bash
cd /path/to/SkillRouter
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
bash scripts/download_eval_data.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_open_models.sh
ls -lt exper_logs | head
```

测试自己训练的 checkpoint：

```bash
cd /path/to/SkillRouter
source .venv/bin/activate
bash scripts/download_eval_data.sh
CUDA_VISIBLE_DEVICES=0 \
SKILLROUTER_EMB_MODEL_OR_PATH=/path/to/your/embedding-checkpoint \
SKILLROUTER_RERANK_MODEL_OR_PATH=/path/to/your/reranker-checkpoint \
bash scripts/evaluate_open_models.sh
ls -lt exper_logs | head
```
