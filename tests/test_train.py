import importlib
import io
import sys
import types
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch

from src.utils import EMA, save_checkpoint


class ResumeWeightsOnlyTests(unittest.TestCase):
    def test_parser_accepts_weights_only_resume_flag(self):
        """Catch the fine-tuning flag being unavailable from the training command."""
        sys.modules.setdefault("wandb", types.ModuleType("wandb"))
        train = importlib.import_module("train")

        with patch.object(sys, "argv", ["train.py", "--resume", "model.pth", "--resume_weights_only"]):
            args = train.parse_args()

        self.assertTrue(args.resume_weights_only)

    def test_weights_only_resume_restores_weights_but_restarts_training_state(self):
        """Catch loading an exhausted optimizer or scheduler during fine-tuning."""
        sys.modules.setdefault("wandb", types.ModuleType("wandb"))
        train = importlib.import_module("train")

        source_model = torch.nn.Linear(2, 1, bias=False)
        source_ema = EMA(source_model)
        source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.25)
        source_scheduler = torch.optim.lr_scheduler.ConstantLR(source_optimizer, factor=1.0, total_iters=10)
        with torch.no_grad():
            source_model.weight.fill_(3.0)
            source_ema.ema_model.weight.fill_(5.0)
        source_optimizer.zero_grad()
        source_model(torch.ones(1, 2)).sum().backward()
        source_optimizer.step()
        source_scheduler.step()
        with torch.no_grad():
            source_model.weight.fill_(3.0)
            source_ema.ema_model.weight.fill_(5.0)

        target_model = torch.nn.Linear(2, 1, bias=False)
        target_ema = EMA(target_model)
        target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.01)
        target_scheduler = torch.optim.lr_scheduler.ConstantLR(target_optimizer, factor=1.0, total_iters=10)

        checkpoint = io.BytesIO()
        save_checkpoint(
            checkpoint, source_model, source_ema, source_optimizer,
            source_scheduler, epoch=99, step=1_000, args={},
        )
        checkpoint.seek(0)
        start_epoch, global_step = train.resume_training_state(
            Namespace(resume=checkpoint, resume_weights_only=True, device="cpu"),
            target_model, target_ema, target_optimizer, target_scheduler,
        )

        torch.testing.assert_close(target_model.weight, torch.full_like(target_model.weight, 3.0))
        torch.testing.assert_close(target_ema.ema_model.weight, torch.full_like(target_ema.ema_model.weight, 5.0))
        self.assertEqual((start_epoch, global_step), (0, 0))
        self.assertEqual(target_optimizer.state, {})
        self.assertEqual(target_scheduler.last_epoch, 0)
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.01)


class DiTOnlyTrainingTests(unittest.TestCase):
    @staticmethod
    def load_train():
        sys.modules.setdefault('wandb', types.ModuleType('wandb'))
        return importlib.import_module('train')

    def test_parser_does_not_expose_adaptive_cfg_configuration(self):
        train = self.load_train()
        with patch.object(sys, 'argv', ['train.py']):
            args = train.parse_args()

        self.assertFalse(hasattr(args, 'adaptive_cfg'))

    def test_preview_rejects_an_adaptive_controller_argument(self):
        train = self.load_train()
        args = Namespace(device='cpu', image_size=2, sampler='ddim',
                         num_sampling_steps=1, cfg_scale=4.0)
        class Diffusion:
            def sample(self, *unused_args, **unused_kwargs):
                return torch.zeros(1, 3, 2, 2)

        with patch.object(train, 'save_image'), self.assertRaises(TypeError):
            train.sample_preview(Diffusion(), object(), 1, args, 'unused.png',
                                 adaptive_cfg=object())