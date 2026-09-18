# AI-MODEL

这是一个用于学习中文 Encoder–Decoder GRU 的小型 PyTorch Lightning 项目。当前结构把数据、分词、Word2Vec、训练、恢复和聊天分开，同时保留原来的网络、损失、Adam 和贪心生成算法。

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

复制并修改 `configs/cloud.yaml`，特别是持久化 `output_dir`、数据规模和硬件参数：

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

默认数据是 Hugging Face `silver/lccc` 的 `base` 配置，读取 `dialog` 字段。使用本地 JSON 时：

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
logs/
manifest.json
records/
```

## 断点续训

再次使用同一输出目录时，程序优先恢复 `last.ckpt`；`max_epochs` 是累计目标轮数：

```bash
python -m ai_model train \
  --config configs/cloud.yaml \
  --output-dir /persistent/ai-model/experiment-001 \
  --max-epochs 1200
```

也可以用 `--resume-from /path/to/file.ckpt` 显式选择文件。选中的文件损坏或不兼容时会直接报错，不会换用另一个 checkpoint。

`--no-resume` 用于新网络权重，并拒绝任何已经包含 checkpoint 的目标目录。新实验请选择新的 `output_dir`；它不会删除旧模型。

## 聊天

```bash
python -m ai_model chat \
  --output-dir /persistent/ai-model/experiment-001 \
  --device cpu
```

输入 `exit`、`quit` 或 `退出` 结束。聊天只读取 checkpoint、词表和 Word2Vec，不访问训练数据，也不会补造缺失产物。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试包括训练核心等价、数据契约、配置与检查点，以及真实 CPU 小数据保存、续训和聊天加载。测试通过证明流程与原计算语义成立；它不证明回答质量，也不等于完成云端 GPU 长训练。

详细调用图、张量形状和架构概念见 [架构与源码阅读手册](docs/architecture-guide.md)，并可直接打开 `docs/architecture-guide.html`。如需重新生成 HTML 和静态导图，先安装 `requirements-docs.txt`，再执行 `python docs/build_guide.py`。
