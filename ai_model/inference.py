"""Greedy response generation using an already loaded training model."""

from time import sleep

import torch

from .tokenizer import Tokenizer
from .training import Train

GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"


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


class ChatSession:
    """Own a model and tokenizer; inference does not load or prepare training data."""

    def __init__(self, model: Train, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

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

        self.model.eval()
        bos_id = self.tokenizer.bos_id
        eos_id = self.tokenizer.eos_id
        model_device = next(self.model.parameters()).device
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
            question_embedding = self.model.embedding(question_ids)
            _, decoder_hidden = self.model.encoder(question_embedding)

            for _ in range(max_new_tokens):
                decoder_embedding = self.model.embedding(current_id)
                decoder_output, decoder_hidden = self.model.decoder(
                    decoder_embedding,
                    decoder_hidden,
                )
                next_id = self.model.linear(decoder_output[:, -1, :]).argmax(
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
        self.model.eval()

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
