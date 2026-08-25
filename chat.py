import argparse
from pathlib import Path
from time import sleep

import torch

from color import BLUE, GREEN, RESET
from config import Config
from data_provider import DataProvider, LoadingMethod
from tokenizer import Tokenizer
from train import Train


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


class Chat(Train):
    def generate(
        self,
        question: str,
        max_new_tokens: int = 50,
    ) -> torch.Tensor:
        """Generate one response, feeding each predicted token back to the GRU."""
        if not question.strip():
            raise ValueError("question cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")

        bos_id = self.tokenizer.bos_id
        eos_id = self.tokenizer.eos_id
        model_device = next(self.parameters()).device
        question_ids = torch.tensor(
            self.tokenizer.get_ids(question),
            dtype=torch.long,
            device=model_device,
        ).unsqueeze(0)

        current_id = torch.full(
            (1, 1),
            fill_value=bos_id,
            dtype=torch.long,
            device=model_device,
        )
        generated_ids = current_id

        with torch.inference_mode():
            question_embedding = self.embedding(question_ids)
            _, decoder_hidden = self.encoder(question_embedding)

            for _ in range(max_new_tokens):
                decoder_embedding = self.embedding(current_id)
                decoder_output, decoder_hidden = self.decoder(
                    decoder_embedding,
                    decoder_hidden,
                )
                next_id = self.linear(decoder_output[:, -1, :]).argmax(
                    dim=-1,
                    keepdim=True,
                )
                generated_ids = torch.cat([generated_ids, next_id], dim=1)
                if next_id.item() == eos_id:
                    break
                current_id = next_id

        return generated_ids

    def begin_chat(
        self,
        max_new_tokens: int = 50,
        typing_delay: float = 0.05,
    ) -> None:
        self.eval()

        while True:
            try:
                question = input(f"{GREEN}You (输入 exit 退出): {RESET}")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.strip().lower() in {"exit", "quit", "退出"}:
                break

            try:
                generated_ids = self.generate(
                    question,
                    max_new_tokens=max_new_tokens,
                )
            except ValueError as error:
                print(f"输入错误: {error}")
                continue

            answer = self.tokenizer.decode(generated_ids[0])
            self.print_answer(answer, typing_delay)

    @staticmethod
    def print_answer(answer: str, typing_delay: float = 0.05) -> None:
        if typing_delay < 0:
            raise ValueError("typing_delay cannot be negative")
        print(f"{BLUE}AI : {RESET}", end="")
        for character in answer:
            print(character, flush=True, end="")
            if typing_delay:
                sleep(typing_delay)
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with a trained AI-MODEL checkpoint."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-name", default="silver/lccc")
    parser.add_argument("--dataset-config", default="base")
    parser.add_argument("--dialog-field", default="dialog")
    parser.add_argument("--max-dialogs", type=int, default=10_000)
    parser.add_argument("--max-sequence-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--typing-delay", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than 0")
    if args.typing_delay < 0:
        raise ValueError("typing_delay cannot be negative")

    output_dir = args.output_dir.expanduser().resolve()
    config = Config(output_dir=output_dir)
    checkpoint_dir = config.checkpoint_dir
    vocabulary_path = config.vocabulary_path
    word2vec_path = config.word2vec_path
    requested_checkpoint = args.checkpoint
    checkpoint_path = (
        requested_checkpoint.expanduser().resolve()
        if requested_checkpoint is not None
        else checkpoint_dir / "last.ckpt"
    )
    for artifact_name, artifact_path in (
        ("checkpoint", checkpoint_path),
        ("vocabulary", vocabulary_path),
        ("Word2Vec model", word2vec_path),
    ):
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Required {artifact_name} does not exist: {artifact_path}"
            )

    data_provider = DataProvider(
        args.dataset_name,
        LoadingMethod.PATH,
        max_dialogs=args.max_dialogs,
        dataset_config=args.dataset_config,
        dialog_field=args.dialog_field,
    )
    tokenizer = Tokenizer(
        data_provider,
        max_sequence_length=args.max_sequence_length,
        vocabulary_path=vocabulary_path,
        word2vec_path=word2vec_path,
        allow_vocabulary_updates=False,
    )
    chat_model = Chat(
        vocab_size=len(tokenizer.token_to_id),
        pad_id=tokenizer.pad_id,
        tokenizer=tokenizer,
    )
    checkpoint_data = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint_data.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict")
    chat_model.load_state_dict(state_dict, strict=True)

    runtime_device = resolve_device(args.device)
    chat_model.to(runtime_device)
    print(f"checkpoint={checkpoint_path}")
    print(f"device={runtime_device}")
    chat_model.begin_chat(
        max_new_tokens=args.max_new_tokens,
        typing_delay=args.typing_delay,
    )


if __name__ == "__main__":
    main()
