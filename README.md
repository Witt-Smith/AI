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

聊天默认加载 `lightning_logs/` 中最近保存的检查点，也可以用 `--checkpoint 路径` 指定云端下载的权重。新版检查点保存词表和类别顺序。每条消息独立处理，输出预测意图和回答。默认从概率最高的 5 个 token 中采样，使用 `--temperature 0` 可复现最高分回复。输入 `/exit` 退出。

从项目根目录构建训练镜像可使用 `docker build -f DockerFile/dockfile -t my-ai .`。
