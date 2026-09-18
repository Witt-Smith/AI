"""Data contracts and artifact stability, without downloading remote datasets."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import jieba
import torch
from gensim.models import Word2Vec

from ai_model.data import DataSource, TrainingDataset
from ai_model.embeddings import EmbeddingStore
from ai_model.tokenizer import Tokenizer


class FakeSplit(list):
    column_names = ["utterances"]

    def select(self, indices):
        return [self[index] for index in indices]


class FakeDatasetDict(dict):
    pass


class DataTextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.previous_jieba_dir = jieba.dt.tmp_dir
        jieba.dt.tmp_dir = str(self.root)

    def tearDown(self):
        jieba.dt.tmp_dir = self.previous_jieba_dir
        self.temporary.cleanup()

    def local_data(self, payload):
        path = self.root / "dialogues.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return DataSource(str(path))

    def make_store(self):
        model = Word2Vec(vector_size=128, min_count=2, workers=1)
        model.build_vocab([["你好", "世界", "好", "我", "你", "你好", "世界", "好", "我", "你"]])
        return EmbeddingStore(model, self.root / "word2vec.bin")

    def make_tokenizer(self, max_length=64):
        return Tokenizer.prepare(self.make_store(), self.root / "vocabulary.json", max_length)

    def test_local_cleaning_order_limit_and_cached_reads(self):
        provider = self.local_data({"train": [
            {"question": " 你 好\n世界 ", "answer": "我\t很 好", "record_id": "one"},
            {"question": "第二问", "answer": "第二答"},
        ], "unknown": [" 未 知 "]})
        self.assertEqual(provider.get_pairs(), [("你好世界", "我很好"), ("第二问", "第二答")])
        self.assertEqual(provider.get_records("unknown")[0]["question"], "未知")
        provider.dataset_path.unlink()
        self.assertEqual(len(provider.get_pairs()), 2)
        self.assertEqual(provider.get_dialogs()[0], ["你好世界", "我很好"])
        self.assertEqual(provider.load_word_data(), [list(jieba.cut(text)) for pair in provider.get_pairs() for text in pair])
        with self.assertRaises(ValueError):
            provider.get_pairs("unknown")

    def test_local_list_payload_and_invalid_fields(self):
        provider = self.local_data([{"question": "第一", "answer": "一"}, {"question": "第二", "answer": "二"}])
        provider.max_dialogs = 1
        self.assertEqual(provider.get_pairs(), [("第一", "一")])
        for payload, error in [
            ({"validation": []}, ValueError),
            ({"train": []}, ValueError),
            ({"train": [{"question": "x", "answer": "  "}]}, ValueError),
            ({"train": "invalid"}, TypeError),
            ({"train": [{"question": "x", "answer": "y", "record_id": "a"},
                        {"question": "z", "answer": "w", "record_id": "a"}]}, ValueError),
        ]:
            with self.subTest(payload=payload), self.assertRaises(error):
                self.local_data(payload).get_pairs()

    def test_remote_lazy_loading_field_mapping_adjacency_and_cache(self):
        remote = types.ModuleType("datasets")
        remote.DatasetDict = FakeDatasetDict
        remote.load_dataset = Mock(return_value=FakeDatasetDict(train=FakeSplit([
            {"utterances": [" A B ", "", " C\nD", "EF"]},
            {"utterances": ["ignored", "row"]},
        ])))
        with patch.dict(sys.modules, {"datasets": remote}):
            provider = DataSource("example/dataset", dataset_config="base", dialog_field="utterances", max_dialogs=1)
            remote.load_dataset.assert_not_called()
            self.assertEqual(provider.get_pairs(), [("AB", "CD"), ("CD", "EF")])
            self.assertEqual(provider.load_word_data(), [["AB"], ["CD"], ["EF"]])
            provider.get_pairs()
            remote.load_dataset.assert_called_once_with("example/dataset", "base")
            with self.assertRaisesRegex(ValueError, "no 'test' split"):
                provider.get_pairs("test")
            with self.assertRaises(TypeError):
                provider.get_records()

    def test_remote_malformed_turn_and_missing_field_fail(self):
        remote = types.ModuleType("datasets")
        remote.DatasetDict = FakeDatasetDict
        remote.load_dataset = Mock(return_value=FakeDatasetDict(train=FakeSplit([{"utterances": ["valid", 123]}])))
        with patch.dict(sys.modules, {"datasets": remote}):
            with self.assertRaisesRegex(TypeError, "must be a string"):
                DataSource("example/dataset", dialog_field="utterances").get_pairs()
            with self.assertRaisesRegex(ValueError, "was not found"):
                DataSource("example/dataset").get_pairs()

    def test_ids_sort_special_tokens_unknown_truncation_and_decode(self):
        tokenizer = self.make_tokenizer(max_length=5)
        self.assertEqual(list(tokenizer.token_to_id)[:4], ["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
        expected_words = sorted(tokenizer.store.word2vec.wv.key_to_index)
        self.assertEqual(list(tokenizer.token_to_id)[4:], expected_words)
        text = "你好世界我你好"
        words = list(jieba.cut(text))[:3]
        self.assertEqual(tokenizer.get_ids(text), [2] + [tokenizer.token_to_id.get(word, 1) for word in words] + [3])
        self.assertEqual(tokenizer.get_ids("unseenword"), [2, 1, 3])
        self.assertEqual(tokenizer.get_ids("  "), [2, 3])
        hello = tokenizer.token_to_id["你好"]
        self.assertEqual(tokenizer.decode(torch.tensor([[0, 2, hello, 1, 3, hello]])), "你好<UNK>")
        with self.assertRaises(TypeError):
            tokenizer.decode([True])
        with self.assertRaises(TypeError):
            tokenizer.get_ids(123)

    def test_existing_vocabulary_is_unchanged_and_new_words_not_appended(self):
        tokenizer = self.make_tokenizer()
        path = tokenizer.vocabulary_path
        before = path.read_bytes()
        vocabulary = dict(tokenizer.token_to_id)
        # The store can contain additional words while an existing ID map stays fixed.
        tokenizer.store.word2vec.build_vocab([["新增", "新增"]], update=True)
        loaded = Tokenizer.prepare(tokenizer.store, path)
        self.assertEqual(loaded.token_to_id, vocabulary)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(Tokenizer.load(tokenizer.store, path).token_to_id, vocabulary)
        with self.assertRaises(FileNotFoundError):
            Tokenizer.prepare(tokenizer.store, self.root / "missing.json", allow_vocabulary_updates=False)
        with self.assertRaises(ValueError):
            Tokenizer.prepare(tokenizer.store, self.root / "new.json", max_sequence_length=2)
        self.assertFalse((self.root / "new.json").exists())

    def test_invalid_vocab_and_artifacts(self):
        store = self.make_store()
        path = self.root / "bad.json"
        for vocabulary in [[], {"<PAD>": 0}, {"<PAD>": False, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3},
                           {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "x": 3}]:
            path.write_text(json.dumps(vocabulary), encoding="utf-8")
            with self.subTest(vocabulary=vocabulary), self.assertRaises((ValueError, TypeError)):
                Tokenizer.load(store, path)
        path.write_text("")
        with self.assertRaises(ValueError):
            Tokenizer.load(store, path)
        with self.assertRaises(FileNotFoundError):
            EmbeddingStore.load(self.root / "missing.bin")
        with self.assertRaises(ValueError):
            EmbeddingStore.load(path)

    def test_batch_tensors_true_lengths_order_and_preencoding(self):
        tokenizer = self.make_tokenizer()
        pairs = [("你好", "你好世界"), ("你好世界我", "好")]
        dataset = TrainingDataset(pairs, tokenizer)
        question, answer, lengths = dataset.collate_fn([dataset[0], dataset[1]])
        expected_q = [tokenizer.get_ids(pair[0]) for pair in pairs]
        expected_a = [tokenizer.get_ids(pair[1]) for pair in pairs]
        self.assertEqual(lengths.tolist(), list(map(len, expected_q)))
        self.assertEqual(question.tolist(), [ids + [0] * (question.shape[1] - len(ids)) for ids in expected_q])
        self.assertEqual(answer.tolist(), [ids + [0] * (answer.shape[1] - len(ids)) for ids in expected_a])
        self.assertEqual(question.dtype, torch.long)
        self.assertEqual(answer.dtype, torch.long)
        self.assertEqual(lengths.dtype, torch.long)
        with patch.object(tokenizer, "get_ids", side_effect=AssertionError("re-encoded")):
            self.assertEqual(len(dataset), 2)
            dataset.collate_fn([dataset[0], dataset[1]])

    def test_vectors_align_and_consume_rng_only_when_requested(self):
        store = self.make_store()
        torch.manual_seed(42)
        before = torch.get_rng_state().clone()
        tokenizer = Tokenizer.prepare(store, self.root / "vocabulary.json")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        vocabulary = tokenizer.token_to_id
        expected = torch.empty(len(vocabulary), 128, dtype=torch.float)
        torch.nn.init.normal_(expected, mean=0.0, std=0.02)
        expected[0].zero_()
        expected[1] = torch.from_numpy(store.word2vec.wv.vectors).mean(dim=0)
        for token, token_id in vocabulary.items():
            if token in store.word2vec.wv:
                expected[token_id] = torch.from_numpy(store.word2vec.wv[token].copy())
        torch.set_rng_state(before)
        actual = tokenizer.get_vector(vocabulary)
        self.assertTrue(torch.equal(expected, actual))
        repeated = tokenizer.get_vector(vocabulary)
        self.assertFalse(torch.equal(actual[2:4], repeated[2:4]))
        self.assertTrue(torch.equal(actual[4:], repeated[4:]))

    def test_embedding_prepare_parameters_and_read_only_reuse(self):
        provider = Mock()
        provider.load_word_data.return_value = [["你好", "你好"], ["世界", "世界"]]
        model = self.make_store().word2vec
        path = self.root / "new.bin"
        with patch("ai_model.embeddings.Word2Vec", return_value=model) as factory:
            result = EmbeddingStore.prepare(provider, path)
            factory.assert_called_once_with(provider.load_word_data.return_value, vector_size=128, min_count=2, workers=1, alpha=0.002, epochs=1000)
        self.assertTrue(path.is_file())
        before = path.read_bytes()
        provider.load_word_data.side_effect = AssertionError("must not read provider")
        reloaded = EmbeddingStore.prepare(provider, path)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(result.word2vec.wv.key_to_index, reloaded.word2vec.wv.key_to_index)
        provider.load_word_data.side_effect = None
        provider.load_word_data.return_value = [["once"]]
        with self.assertRaisesRegex(ValueError, "no token appears"):
            EmbeddingStore.prepare(provider, self.root / "insufficient.bin")
        self.assertFalse((self.root / "insufficient.bin").exists())


if __name__ == "__main__":
    unittest.main()
