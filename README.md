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
