"""A real CPU save/resume/chat lifecycle; this intentionally avoids fast_dev_run."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from ai_model.checkpoints import RunPaths, artifact_fingerprints, load_checkpoint
from ai_model.config import AppConfig, ChatConfig, DataConfig, TrainConfig
from ai_model.runtime import run_train


class RuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dialogues.json"
        repeated = [
            {"question": "你好你好", "answer": "你好世界"},
            {"question": "你好世界", "answer": "世界你好"},
            {"question": "世界世界", "answer": "你好你好"},
            {"question": "你好你好", "answer": "世界世界"},
        ]
        self.dataset.write_text(
            json.dumps({"train": repeated}, ensure_ascii=False),
            encoding="utf-8",
        )
        self.output = self.root / "experiment"
        self.config = AppConfig(
            command="train",
            output_dir=self.output,
            data=DataConfig(
                dataset_name=str(self.dataset),
                dataset_config=None,
                dialog_field="dialog",
                max_dialogs=4,
                max_sequence_length=8,
            ),
            train=TrainConfig(
                batch_size=2,
                num_workers=0,
                max_epochs=1,
                learning_rate=0.002,
                seed=123,
                checkpoint_every_n_epochs=1,
                log_every_n_steps=1,
                gradient_clip_val=1.0,
                accelerator="cpu",
                devices=1,
                precision="32-true",
            ),
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def optimizer_step(checkpoint):
        states = checkpoint["optimizer_states"][0]["state"].values()
        return max(int(state["step"].item()) for state in states if "step" in state)

    def test_real_save_resume_and_chat_load(self):
        run_train(self.config)
        paths = RunPaths(self.output)
        self.assertTrue((paths.checkpoint_dir / "last.ckpt").is_file())
        self.assertTrue(paths.manifest_path.is_file())
        first = load_checkpoint(paths.checkpoint_dir / "last.ckpt")
        first_global_step = first["global_step"]
        first_optimizer_step = self.optimizer_step(first)
        first_embedding = first["state_dict"]["embedding.weight"].clone()
        self.assertGreater(first_global_step, 0)

        resumed = replace(
            self.config,
            train=replace(self.config.train, max_epochs=2),
            explicit=frozenset({"train.max_epochs"}),
        )
        run_train(resumed)
        second = load_checkpoint(paths.checkpoint_dir / "last.ckpt")
        self.assertGreater(second["global_step"], first_global_step)
        self.assertGreater(self.optimizer_step(second), first_optimizer_step)
        torch.testing.assert_close(
            second["state_dict"]["embedding.weight"],
            first_embedding,
            rtol=0,
            atol=0,
        )

        before_chat = artifact_fingerprints(paths)
        project_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(project_root)
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "ai_model",
                "chat",
                "--output-dir",
                str(self.output),
                "--device",
                "cpu",
                "--typing-delay",
                "0",
            ],
            cwd=project_root,
            env=environment,
            input="exit\n",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("checkpoint=", completed.stdout)
        self.assertIn("device=cpu", completed.stdout)
        self.assertEqual(artifact_fingerprints(paths), before_chat)


if __name__ == "__main__":
    unittest.main()
