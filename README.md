# 1.本地数据集目前仅支持JSON,格式如下:
```json
{
  "version": "1.0",
  "data": {
    "train": [
      {
        "id": 0,
        "feature": "你好！",
        "label": "你好呀！",
        "category": "greeting"
      }
    ],
    "validation": [
      {
        "id": 1,
        "feature": "你是谁？",
        "label": "我是AI助手。",
        "category": "identity"
      }
    ],
    "test": [
      {
        "id": 2,
        "feature": "你能做什么？",
        "label": "我可以回答问题。",
        "category": "ability"
      }
    ]
  }
}
```

## 训练和聊天

在项目根目录运行以下命令。首次使用分词器会下载 `cl100k_base`。

```sh
conda activate AI
python main.py
python -m chat --prompt "你是谁？"
python -m chat
```

训练使用本地 JSON 中的 `feature`、`label` 和 `category`。只有训练数据时，程序按类别留出约各 10% 的不同问题用于验证和测试，同一问题的多种回答一起划分。提供了验证集或测试集时使用原划分。意图类别和词表由训练集生成。

训练参数只对本次运行生效。直接运行 `python main.py` 会训练 50 轮，自动选择设备，并从 `Config/config.yaml` 读取数据路径、批大小和数据加载进程数。

```sh
# 用 CPU 各跑一个训练、验证和测试批次，检查整条流程
python main.py --fast-dev-run --accelerator cpu --devices 1 --batch-size 8

# 指定本地数据，训练 10 轮，并跳过训练后的测试
python main.py --data-path ./TrainDataSet/test.json --epochs 10 --batch-size 32 --seed 42 --skip-test
```

`--data-path` 的相对路径从运行命令的目录算起。`--num-workers 0` 在主进程加载数据，正整数指定每个加载器的工作进程数。`--devices` 接受设备数量或 `auto`。`--fast-dev-run` 仅检查有数据的阶段，不保存检查点。运行 `python main.py --help` 可查看全部参数。

聊天默认加载 `lightning_logs/` 中最近保存的检查点，也可以用 `--checkpoint 路径` 指定云端下载的权重。新版检查点保存词表和类别顺序。每条消息独立处理，输出预测意图和回答。默认从概率最高的 5 个 token 中采样，使用 `--temperature 0` 可复现最高分回复。输入 `/exit` 退出。

从项目根目录构建训练镜像可使用 `docker build -f DockerFile/dockfile -t my-ai .`。
