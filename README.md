# AI-MODEL

这是一个用于学习中文 Encoder–Decoder GRU 的小型 PyTorch Lightning 项目。当前结构把数据、分词、Word2Vec、训练、验证、恢复和聊天分开，保持原网络参数与 checkpoint 兼容，并在贪心生成时屏蔽无效的 PAD/BOS 输出。

## 安装

Python 3.9 环境中执行：

```bash
python -m pip install -r requirements.txt
```

在项目根目录查看命令：

```bash
python -m ai_model train --help
python -m ai_model chat --help
```

## 云端首次训练

复制并修改 `configs/cloud.yaml`，特别是持久化 `output_dir`、数据规模和硬件参数。默认远程数据以流式方式只读取配置的前 `max_dialogs` 条，不会为了这个上限先下载完整 LCCC：

```bash
python -m ai_model train --config configs/cloud.yaml
```

命令行中明确给出的参数覆盖 YAML，例如：

```bash
python -m ai_model train \
  --config configs/cloud.yaml \
  --output-dir /persistent/ai-model/experiment-002 \
  --batch-size 32 \
  --num-workers 8 \
  --accelerator gpu \
  --devices 1
```

默认数据是 Hugging Face `silver/lccc` 的 `base` 配置，读取 `dialog` 字段。远程数据通过 Hugging Face 的 Parquet 转换分支读取，不执行数据仓库中的 Python 加载脚本；示例云配置把 `dataset_revision` 固定到已验证的 Parquet commit。使用本地 JSON 时：

```bash
python -m ai_model train \
  --dataset-name examples/dialogues.json \
  --output-dir runs/local-demo \
  --max-epochs 1
```

一个输出目录对应一套实验，其中包含：

```text
artifacts/vocabulary.json
artifacts/word2vec.model
checkpoints/last.ckpt
checkpoints/best.ckpt       # 启用 validation_dialogs 时生成
logs/
logs/app.log              # Python logging 运行事件；超出 5 MB 时轮转，保留 3 份
logs/train/version_*/metrics.csv  # Lightning 训练与验证指标
manifest.json
records/
```

## 断点续训

再次使用同一输出目录时，训练优先恢复 `last.ckpt`；`max_epochs` 是累计目标轮数。manifest 会记录规范化训练/验证问答的内容指纹；即使文件路径没变，只要内容变了，续训也会拒绝混用：

```bash
python -m ai_model train \
  --config configs/cloud.yaml \
  --output-dir /persistent/ai-model/experiment-001 \
  --max-epochs 100
```

也可以用 `--resume-from /path/to/file.ckpt` 显式选择文件。选中的文件损坏或不兼容时会直接报错，不会换用另一个 checkpoint。

`--no-resume` 用于新网络权重，并拒绝任何已经包含 checkpoint 的目标目录。新实验请选择新的 `output_dir`；它不会删除旧模型。更换数据集 revision、样本数量、验证集或词表上限也应使用新的目录。

`validation_dialogs > 0` 时会读取验证 split、记录 `validation_loss` 并维护稳定的 `best.ckpt`。这能发现训练集过拟合趋势，但仍不是聊天质量评测。`max_vocabulary_size` 用来约束输出层和 `[batch, time, vocabulary]` logits 的显存规模。

训练启动摘要会分别打印训练/验证问题与答案的 `unknown_token_ratio`；任一比例超过 10% 会明确警告。高比例意味着大量真实目标被压成 `<UNK>`，此时继续堆 epoch 往往只会让模型更擅长输出 `<UNK>`，应先检查数据量、分词、Word2Vec `MIN_COUNT` 和词表上限。训练和聊天的运行事件通过 Python `logging` 同时写入终端与 `logs/app.log`；训练损失与学习率仍写入 Lightning 的 `metrics.csv`。聊天日志不记录用户输入或模型回答。

## 聊天

```bash
python -m ai_model chat \
  --output-dir /persistent/ai-model/experiment-001 \
  --device cpu
```

输入 `exit`、`quit` 或 `退出` 结束。聊天优先读取 `best.ckpt`，没有验证模型时再读取 `last.ckpt`；它只读取 checkpoint、词表和 Word2Vec，不访问训练数据，也不会补造缺失产物。生成时 PAD/BOS 会被屏蔽，EOS 或 `max_new_tokens` 负责停止。

训练和聊天的成功运行只证明工程链路可以执行；不证明对改写问题的泛化、事实性或真实聊天质量，也不等于完成云端 GPU 长训练。
