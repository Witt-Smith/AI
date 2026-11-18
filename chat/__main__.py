import argparse
import sys
from math import isfinite
from collections.abc import Iterator
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    # 直接运行此文件时，Python 从 chat/ 查找模块，需要加上同级包所在的目录。
    sys.path.insert(0, str(PROJECT_ROOT))

from Data.dataset import DataSet
from Model.network import NeuralNetwork
from Tokenizer.tokenizer import Tokenizer


LOG_DIR = PROJECT_ROOT / "lightning_logs"


def latest_checkpoint(directory: Path) -> Path:
    files = (path for path in directory.rglob("*.ckpt") if path.is_file())
    latest = max(files, key=lambda path: (path.stat().st_mtime_ns, path), default=None)
    if latest is None:
        raise FileNotFoundError(f"在 {directory} 中找不到 .ckpt 文件，可用 --checkpoint 指定。")
    return latest


def load_for_chat(path: Path) -> tuple[NeuralNetwork, Tokenizer]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到检查点 {path}")
    tokenizer = Tokenizer()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    params = checkpoint.get("hyper_parameters", {})
    weights = checkpoint["state_dict"]
    if params.get("vocabulary") is not None:
        tokenizer.token_tale = {
            token: i for i, token in enumerate(params["vocabulary"], tokenizer.SPECIAL_TOKEN_COUNT)
        }
    else:
        # 旧版检查点需要原始数据和当时的遍历顺序。
        tokenizer.assign_ids(DataSet(tokenizer).ds)
    model = NeuralNetwork(
        len(tokenizer), intent_names=params.get("intent_names", ("identity", "ability", "unknown")),
        max_seq_len=weights["position_embedding.weight"].size(0),
    )
    model.load_state_dict(weights)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def answer_question(
    model: NeuralNetwork, tokenizer: Tokenizer, question: str, max_new_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    if max_new_tokens < 1 or not isfinite(temperature) or temperature < 0:
        raise ValueError("生成长度必须大于 0，temperature 必须是有限的非负数。")
    # 空回答的编码以 SEP、EOS 结尾。保留 SEP，让它预测第一个回答 token。
    context = tokenizer.encode(question, "")[:-1].unsqueeze(0).to(model.device)
    if context.size(1) > model.max_seq_len:
        raise ValueError(
            f"问题编码后有 {context.size(1)} 个 token，超过上限 {model.max_seq_len}。"
        )
    reply_ids = []
    intent_name = ""
    for step in range(max_new_tokens):
        logits, intent = model(context)
        if step == 0:
            intent_name = model.intent_names[int(intent.argmax(dim=1).item())]
        scores = logits[:, -1, :].clone()
        scores[:, [tokenizer.pad, tokenizer.bos, tokenizer.sep, tokenizer.unk]] = -torch.inf
        if temperature == 0:
            next_token = scores.argmax(dim=-1, keepdim=True)
        else:
            values, indices = scores.topk(min(5, scores.size(-1)))
            probabilities = torch.softmax((values - values[:, :1]) / temperature, dim=-1)
            next_token = indices.gather(1, torch.multinomial(probabilities, 1))
        token_id = int(next_token.item())
        if token_id == tokenizer.eos:
            break
        reply_ids.append(token_id)
        if context.size(1) == model.max_seq_len:
            break
        context = torch.cat((context, next_token), dim=1)
    return tokenizer.decode(reply_ids), intent_name


def generate_reply(
    model: NeuralNetwork, tokenizer: Tokenizer, question: str, max_new_tokens: int
) -> str:
    reply, _ = answer_question(model, tokenizer, question, max_new_tokens, temperature=0)
    return reply


def interactive_questions() -> Iterator[str]:
    print("每条消息独立测试，输入 exit 或 /exit 退出。")
    while True:
        question = input("你：").strip()
        if question.lower() in ("exit", "/exit"):
            return
        if question:
            yield question


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Lightning 检查点测试单轮问答。")
    parser.add_argument("--checkpoint", type=Path, help="检查点路径，默认选择最近保存的 .ckpt")
    parser.add_argument("--prompt", help="只回答这一句，然后退出")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="最多生成的 token 数，默认 32")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="从前 5 个候选采样，0 表示总是选最高分，默认 0.8")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens 必须大于 0")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt 不能为空")

    try:
        path = args.checkpoint.expanduser() if args.checkpoint else latest_checkpoint(LOG_DIR)
        model, tokenizer = load_for_chat(path)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        parser.exit(1, f"加载失败：{error}\n")
    print(f"已加载 {path}")
    questions = interactive_questions() if args.prompt is None else [args.prompt.strip()]
    try:
        for question in questions:
            try:
                reply, intent = answer_question(model, tokenizer, question, args.max_new_tokens, args.temperature)
            except ValueError as error:
                if args.prompt is not None:
                    parser.exit(1, f"{error}\n")
                print(error)
                continue
            print(f"意图：{intent}")
            print(f"模型：{reply or '（直接生成了 EOS，回复为空）'}")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
