"""Configuration, checkpoint-selection and experiment-provenance contracts."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from ai_model.checkpoints import (
    RunPaths,
    artifact_fingerprints,
    check_checkpoint_origin,
    check_manifest,
    load_checkpoint,
    read_manifest,
    resolve_checkpoint,
    validate_checkpoint,
    write_run_record,
)
from ai_model.config import load_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_defaults_yaml_and_cli_precedence_with_relative_paths(self):
        data = self.root / "dialogues.json"
        data.write_text('{"train": [{"question": "q", "answer": "a"}]}')
        config_path = self.root / "cloud.yaml"
        config_path.write_text(
            "output_dir: runs/one\n"
            "data:\n"
            "  dataset_name: dialogues.json\n"
            "  max_dialogs: 12\n"
            "train:\n"
            "  batch_size: 16\n",
            encoding="utf-8",
        )
        config = load_config(
            "train",
            config_path,
            {"train.batch_size": 8, "output_dir": self.root / "cli-run"},
        )
        self.assertEqual(config.output_dir, (self.root / "cli-run").resolve())
        self.assertEqual(config.data.dataset_name, str(data.resolve()))
        self.assertEqual(config.data.max_dialogs, 12)
        self.assertEqual(config.train.batch_size, 8)
        self.assertEqual(config.train.num_workers, 0)
        self.assertIn("train.batch_size", config.explicit)
        self.assertIn("data.max_dialogs", config.explicit)

    def test_unknown_invalid_and_mutually_exclusive_settings_fail(self):
        for body, message in (
            ("unknown: true\n", "Unknown"),
            ("train:\n  missing: 1\n", "Unknown"),
            ("train:\n  batch_size: 0\n", "batch_size"),
            ("data:\n  max_sequence_length: 2\n", "max_sequence_length"),
            ("train:\n  max_time: 99:99:99:99\n", "max_time"),
        ):
            with self.subTest(body=body):
                path = self.root / "bad.yaml"
                path.write_text(body, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_config("train", path)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            load_config(
                "train",
                overrides={
                    "train.resume_from": self.root / "one.ckpt",
                    "train.no_resume": True,
                },
            )


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = RunPaths(self.root / "run")
        self.paths.prepare()

    def tearDown(self):
        self.temporary.cleanup()

    def touch(self, name, contents=b"checkpoint"):
        path = self.paths.checkpoint_dir / name
        path.write_bytes(contents)
        return path

    def test_selection_order_explicit_last_latest_and_no_resume(self):
        first = self.touch("epoch-1.ckpt")
        second = self.touch("epoch-2.ckpt")
        first.touch()
        second.touch()
        self.assertEqual(resolve_checkpoint(None, False, self.paths.checkpoint_dir), second)
        last = self.touch("last.ckpt")
        self.assertEqual(resolve_checkpoint(None, False, self.paths.checkpoint_dir), last)
        explicit = self.root / "external.ckpt"
        explicit.write_bytes(b"external")
        self.assertEqual(
            resolve_checkpoint(explicit, False, self.paths.checkpoint_dir),
            explicit.resolve(),
        )
        with self.assertRaisesRegex(ValueError, "refuses existing"):
            resolve_checkpoint(None, True, self.paths.checkpoint_dir)
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(self.root / "missing.ckpt", False, self.paths.checkpoint_dir)

    def test_corrupt_checkpoint_does_not_fall_back(self):
        corrupt = self.touch("last.ckpt", b"not a torch checkpoint")
        self.touch("epoch-1.ckpt", b"also not relevant")
        with self.assertRaisesRegex(ValueError, str(corrupt)):
            load_checkpoint(corrupt)

    def test_manifest_records_and_detects_artifact_change(self):
        self.paths.vocabulary_path.write_text('{"<PAD>": 0}', encoding="utf-8")
        self.paths.word2vec_path.write_bytes(b"word-vectors")
        hashes = artifact_fingerprints(self.paths)
        config = load_config("train", overrides={"output_dir": self.paths.output_dir})
        record = write_run_record(self.paths, config, hashes, None)
        self.assertTrue(record.is_file())
        manifest = read_manifest(self.paths)
        self.assertEqual(manifest["artifacts"], hashes)
        check_manifest(self.paths, manifest)
        self.paths.vocabulary_path.write_text('{"<PAD>": 1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Artifacts changed"):
            check_manifest(self.paths, manifest)

    def test_checkpoint_origin_and_shapes_are_strict(self):
        class Model:
            hparams = {"vocab_size": 2, "pad_id": 0}

            @staticmethod
            def state_dict():
                return {"linear.weight": torch.zeros(2, 3)}

        validate_checkpoint(
            {
                "state_dict": {"linear.weight": torch.ones(2, 3)},
                "hyper_parameters": {"vocab_size": 2, "pad_id": 0},
            },
            Model(),
        )
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            validate_checkpoint(
                {"state_dict": {"linear.weight": torch.ones(3, 2)}},
                Model(),
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_checkpoint_origin(self.root / "legacy.ckpt", {"vocabulary.json": "x"})
        self.assertRegex(str(caught[0].message), "same-size token reordering")


if __name__ == "__main__":
    unittest.main()
