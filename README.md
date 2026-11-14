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

## 测试聊天

在项目根目录运行 `python -m chat`，输入问题后按回车，输入 `/exit` 退出。程序按修改时间加载 `lightning_logs/` 下最新的 `.ckpt`。这份检查点需要原训练数据及其顺序来还原词表。

```sh
conda activate AI
python -m chat
python -m chat --prompt "你是谁？" --max-new-tokens 8
```

可以用 `--checkpoint 路径` 指定权重。每次提问都从该问题开始，逐步选择得分最高的 token，遇到 EOS 或长度上限时结束。
