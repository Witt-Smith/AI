"""Regression evidence for the unchanged Train class and legacy greedy decoding."""

import ast
from contextlib import redirect_stdout
import inspect
import io
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import lightning as L
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from ai_model.inference import ChatSession, resolve_device
from ai_model.training import Train

FIXTURES = Path(__file__).parent / "fixtures"


def baseline_class():
    """Execute the pre-move class with only its original external names supplied."""
    namespace = {
        "__name__": __name__,
        "L": L,
        "torch": torch,
        "nn": nn,
        "pack_padded_sequence": pack_padded_sequence,
        "Tokenizer": object,
    }
    path = FIXTURES / "train_class_before.py.txt"
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace["Train"]


def legacy_generate():
    namespace = {"torch": torch}
    path = FIXTURES / "legacy_generate.py.txt"
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace["generate"]


class TinyTokenizer:
    """A fixed, small vocabulary: no data provider, artifact writes, or downloads."""

    token_to_id = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
                   "你": 4, "好": 5, "再": 6, "见": 7}
    pad_id, unk_id, bos_id, eos_id = 0, 1, 2, 3

    def __init__(self):
        self.vector_calls = 0

    def get_vector(self, vocabulary):
        self.vector_calls += 1
        # Deliberately consume RNG so construction-order regressions are visible.
        vectors = torch.empty(len(vocabulary), 5)
        nn.init.normal_(vectors, mean=0, std=0.02)
        vectors[self.pad_id].zero_()
        return vectors

    def get_ids(self, text):
        return [self.bos_id, *[self.token_to_id.get(c, self.unk_id)
                              for c in text.strip()], self.eos_id]

    def decode(self, ids):
        mapping = {value: key for key, value in self.token_to_id.items()}
        result = []
        for token_id in ids.tolist():
            if token_id == self.eos_id:
                break
            if token_id not in (self.pad_id, self.bos_id):
                result.append(mapping.get(token_id, "<UNK>"))
        return "".join(result)


def make_model(model_class=Train, seed=17):
    torch.manual_seed(seed)
    tokenizer = TinyTokenizer()
    model = model_class(8, tokenizer.pad_id, tokenizer, learning_rate=0.002)
    return model, tokenizer


class CorePreservationTests(unittest.TestCase):
    def test_train_class_ast_identical(self):
        before = ast.parse((FIXTURES / "train_class_before.py.txt").read_text()).body[0]
        after = ast.parse(inspect.getsource(Train)).body[0]
        self.assertEqual(ast.dump(before, include_attributes=False),
                         ast.dump(after, include_attributes=False))

    def test_same_seed_initialization_keys_hparams_and_frozen_embedding(self):
        original, old_tokenizer = make_model(baseline_class())
        current, new_tokenizer = make_model()
        expected_keys = ["embedding.weight", "linear.weight", "linear.bias",
                         "encoder.weight_ih_l0", "encoder.weight_hh_l0",
                         "encoder.bias_ih_l0", "encoder.bias_hh_l0",
                         "decoder.weight_ih_l0", "decoder.weight_hh_l0",
                         "decoder.bias_ih_l0", "decoder.bias_hh_l0"]
        self.assertEqual(list(current.state_dict()), expected_keys)
        self.assertEqual(list(current.state_dict()), list(original.state_dict()))
        self.assertEqual(dict(current.hparams), {"vocab_size": 8, "pad_id": 0,
                                                "learning_rate": 0.002})
        self.assertEqual(dict(current.hparams), dict(original.hparams))
        for key, value in current.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(old_tokenizer.vector_calls, 1)
        self.assertEqual(new_tokenizer.vector_calls, 1)
        self.assertFalse(current.embedding.weight.requires_grad)
        self.assertEqual(current.embedding.padding_idx, 0)
        self.assertEqual(current.loss_fn.ignore_index, 0)

    def test_logits_loss_gradients_and_one_adam_update_identical(self):
        original, _ = make_model(baseline_class())
        current, _ = make_model(seed=123)
        current.load_state_dict(original.state_dict(), strict=True)
        questions = torch.tensor([[2, 4, 5, 3], [2, 5, 3, 0]])
        answers = torch.tensor([[2, 6, 7, 3, 0], [2, 4, 5, 7, 3]])
        lengths = torch.tensor([4, 3])
        batch = (questions, answers, lengths)
        before_embedding = current.embedding.weight.detach().clone()
        logits = []
        losses = []
        optimizers = []
        logs = []
        for model in (original, current):
            optimizer = model.configure_optimizers()
            self.assertIsInstance(optimizer, torch.optim.Adam)
            logits.append(model(questions, answers[:, :-1], lengths))
            with patch.object(model, "optimizers", return_value=optimizer), \
                    patch.object(model, "log_dict") as log:
                loss = model.training_step(batch, 0)
                logs.append(log.call_args)
            self.assertEqual(set(logs[-1].args[0]), {"normal_loss", "lr"})
            self.assertEqual(logs[-1].args[0]["lr"], 0.002)
            self.assertEqual(logs[-1].kwargs, {"on_step": False, "on_epoch": True,
                                             "prog_bar": True, "logger": True,
                                             "batch_size": 2})
            losses.append(loss)
            loss.backward()
            optimizers.append(optimizer)
        self.assertEqual(tuple(logits[0].shape), (2, 4, 8))
        torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        for (name, old_parameter), (new_name, parameter) in zip(
                original.named_parameters(), current.named_parameters()):
            self.assertEqual(name, new_name)
            if name == "embedding.weight":
                self.assertIsNone(parameter.grad)
                self.assertIsNone(old_parameter.grad)
            else:
                self.assertIsNotNone(parameter.grad)
                torch.testing.assert_close(old_parameter.grad, parameter.grad, rtol=0, atol=0)
        for optimizer in optimizers:
            optimizer.step()
        for key, value in current.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(current.embedding.weight, before_embedding, rtol=0, atol=0)
        self.assertTrue(optimizers[1].state)
        for parameter, state in optimizers[1].state.items():
            self.assertEqual(state["step"].item(), 1)
            self.assertIn("exp_avg", state)
            self.assertIn("exp_avg_sq", state)

    def test_constructor_rejects_invalid_contract(self):
        for vocab_size, pad_id, learning_rate in ((9, 0, .002), (8, -1, .002),
                                                   (8, 8, .002), (8, 0, 0)):
            with self.subTest(vocab_size=vocab_size, pad_id=pad_id, lr=learning_rate):
                with self.assertRaises(ValueError):
                    Train(vocab_size, pad_id, TinyTokenizer(), learning_rate)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.model, self.tokenizer = make_model()
        self.session = ChatSession(self.model, self.tokenizer)

    def force_prediction(self, token_id):
        with torch.no_grad():
            self.model.linear.weight.zero_()
            self.model.linear.bias.fill_(-10)
            self.model.linear.bias[token_id] = 10

    def test_ids_match_history_for_known_and_unknown_questions(self):
        old_generate = types.MethodType(legacy_generate(), self.model)
        for question in ("你好", "陌", "再见"):
            with self.subTest(question=question):
                self.model.eval()
                before = old_generate(question, max_new_tokens=8)
                after = self.session.generate(question, max_new_tokens=8)
                torch.testing.assert_close(after, before, rtol=0, atol=0)
                self.assertEqual(after[0, 0].item(), self.tokenizer.bos_id)
                self.assertFalse(after.requires_grad)
        self.assertEqual(self.tokenizer.get_ids("陌"), [2, 1, 3])

    def test_eos_stops_immediately_and_question_encoded_once(self):
        self.force_prediction(self.tokenizer.eos_id)
        self.model.train()
        calls = []
        hook = self.model.encoder.register_forward_hook(
            lambda module, arguments, output: calls.append(torch.is_inference_mode_enabled()))
        try:
            with patch.object(self.tokenizer, "get_ids", wraps=self.tokenizer.get_ids) as encode:
                result = self.session.generate("你好", max_new_tokens=7)
                encode.assert_called_once_with("你好")
        finally:
            hook.remove()
        self.assertEqual(result.tolist(), [[2, 3]])
        self.assertEqual(calls, [True])
        self.assertFalse(self.model.training)

    def test_length_limit_keeps_bos_and_each_greedy_token(self):
        self.force_prediction(5)
        calls = []
        hook = self.model.encoder.register_forward_hook(
            lambda module, arguments, output: calls.append(output))
        try:
            generated = self.session.generate("你好", max_new_tokens=4)
        finally:
            hook.remove()
        self.assertEqual(generated.tolist(), [[2, 5, 5, 5, 5]])
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.tokenizer.decode(generated[0]), "好好好好")

    def test_blank_input_and_invalid_length(self):
        for question, limit in (("", 4), (" \t\n", 4), ("你好", 0), ("你好", -1)):
            with self.subTest(question=question, limit=limit):
                with self.assertRaises(ValueError):
                    self.session.generate(question, limit)

    def test_all_exit_words_and_eof_interrupt(self):
        for stopping_input in ("exit", " EXIT ", "quit", "退出", EOFError, KeyboardInterrupt):
            with self.subTest(stopping_input=stopping_input):
                with patch("builtins.input", side_effect=[stopping_input]), \
                        patch.object(self.session, "generate") as generate, \
                        redirect_stdout(io.StringIO()):
                    self.session.begin_chat(typing_delay=0)
                generate.assert_not_called()

    def test_interactive_blank_then_answer_then_exit(self):
        self.force_prediction(5)
        output = io.StringIO()
        with patch("builtins.input", side_effect=[" ", "你好", "exit"]), redirect_stdout(output):
            self.session.begin_chat(max_new_tokens=2, typing_delay=0)
        self.assertIn("输入错误: question cannot be empty", output.getvalue())
        self.assertIn("好好", output.getvalue())

    def test_printing_delay_and_error_match_original_behavior(self):
        with redirect_stdout(io.StringIO()), patch("ai_model.inference.sleep") as sleep:
            self.session.print_answer("你好", typing_delay=0)
            sleep.assert_not_called()
            self.session.print_answer("你好", typing_delay=0.01)
            self.assertEqual(sleep.call_count, 2)
            sleep.assert_called_with(0.01)
            with self.assertRaisesRegex(ValueError, "typing_delay"):
                self.session.print_answer("你好", typing_delay=-1)

    def test_auto_device_priority_and_unavailable_devices(self):
        for cuda_available, mps_available, expected in ((True, True, "cuda"),
                                                       (False, True, "mps"),
                                                       (False, False, "cpu")):
            with self.subTest(cuda=cuda_available, mps=mps_available), \
                    patch("torch.cuda.is_available", return_value=cuda_available), \
                    patch("torch.backends.mps.is_available", return_value=mps_available):
                self.assertEqual(str(resolve_device("auto")), expected)
                self.assertEqual(str(resolve_device("cpu")), "cpu")
                if not cuda_available:
                    with self.assertRaisesRegex(ValueError, "CUDA"):
                        resolve_device("cuda")
                if not mps_available:
                    with self.assertRaisesRegex(ValueError, "MPS"):
                        resolve_device("mps")


if __name__ == "__main__":
    unittest.main()
