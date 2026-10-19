import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from src.adaptive_cfg import AdaptiveCFG
from src.diffusion import GaussianDiffusion
from src.model import DiT


class PredictionModel(nn.Module):
    num_classes = 2

    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.25))

    def forward(self, images, timestep, labels):
        del timestep
        return (self.offset + labels.reshape(-1, 1, 1, 1).to(images.dtype)).expand_as(images)


class AdaptiveCFGTrainerTests(unittest.TestCase):
    def test_save_cfg_curve_writes_a_timestep_curve_image(self):
        import train_cfg

        controller = AdaptiveCFG(10, hidden_size=4)
        with tempfile.TemporaryDirectory() as directory:
            path = train_cfg.save_cfg_curve(controller, 10, 3, directory, 'cpu')

            self.assertEqual(path, Path(directory) / 'cfg_curves' / 'epoch_003.png')
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_save_epoch_preview_uses_controller_and_preserves_rng_state(self):
        import train_cfg

        class Diffusion:
            def __init__(self):
                self.received_controller = None

            def sample(self, model, num_images, labels, image_size, image_channel,
                       sampler, num_sampling_steps, cfg_scale, device, adaptive_cfg):
                del model, image_size, image_channel, sampler, num_sampling_steps, cfg_scale, device
                self.received_controller = adaptive_cfg
                torch.rand(1)
                self.labels = labels.detach().cpu()
                return torch.zeros(num_images, 3, 2, 2)

        args = Namespace(
            device='cpu', preview_sampler='ddim', preview_num_sampling_steps=4,
            preview_num_images=2, preview_seed=77,
        )
        diffusion = Diffusion()
        controller = AdaptiveCFG(10, hidden_size=4)
        torch.manual_seed(123)
        expected_after_preview = torch.rand(1)
        torch.manual_seed(123)

        with tempfile.TemporaryDirectory() as directory:
            path = train_cfg.save_epoch_preview(
                diffusion, object(), controller, 10, 2, args, 3, directory,
            )

            self.assertEqual(path, Path(directory) / 'cfg_previews' / 'epoch_003.png')
            self.assertTrue(path.is_file())
        self.assertIs(diffusion.received_controller, controller)
        torch.testing.assert_close(diffusion.labels, torch.tensor([0, 1]))
        torch.testing.assert_close(torch.rand(1), expected_after_preview)

    def test_build_frozen_ema_model_loads_ema_weights_without_gradients(self):
        import train_cfg

        source_model = DiT(
            image_size=4, patch_size=2, hidden_size=8, depth=1,
            num_heads=2, num_classes=10,
        )
        with torch.no_grad():
            source_model.pos_embed.fill_(0.75)
        checkpoint = {
            'args': {
                'dataset': 'cifar10', 'image_size': 4, 'patch_size': 2,
                'hidden_size': 8, 'depth': 1, 'num_heads': 2,
                'mlp_ratio': 4.0, 'class_dropout_prob': 0.1,
            },
            'ema_state_dict': source_model.state_dict(),
        }

        frozen = train_cfg.build_frozen_ema_model(checkpoint, 'cpu')

        self.assertFalse(frozen.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen.parameters()))
        torch.testing.assert_close(frozen.pos_embed, torch.full_like(frozen.pos_embed, 0.75))

    def test_parse_args_requires_checkpoint_and_output(self):
        import train_cfg

        with patch.object(sys, 'argv', [
                'train_cfg.py', '--checkpoint', 'base.pth', '--output', 'cfg.pth']):
            args = train_cfg.parse_args()

        self.assertEqual(args.checkpoint, 'base.pth')
        self.assertEqual(args.output, 'cfg.pth')
        self.assertEqual(args.save_every, 10)

    def test_controller_step_leaves_frozen_dit_unchanged(self):
        import train_cfg

        model = PredictionModel().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        controller = AdaptiveCFG(10, hidden_size=4)
        optimizer = torch.optim.AdamW(controller.parameters(), lr=1e-2)
        diffusion = GaussianDiffusion(10, beta_schedule='cosine')
        classifier = nn.Sequential(nn.Flatten(), nn.Linear(4, 2)).eval().requires_grad_(False)
        classifier_before = [p.detach().clone() for p in classifier.parameters()]
        images = torch.zeros(2, 1, 2, 2)
        labels = torch.tensor([0, 1])
        model_before = [parameter.detach().clone() for parameter in model.parameters()]
        controller_before = [parameter.detach().clone() for parameter in controller.parameters()]

        total, _, _ = train_cfg.train_controller_step(
            diffusion, model, controller, optimizer, images, labels,
            classifier=classifier, classification_weight=0.01,
            classification_min_snr=0.0,
        )

        self.assertTrue(torch.isfinite(total))
        self.assertTrue(all(p.grad is None and torch.equal(p, before)
                            for p, before in zip(classifier.parameters(), classifier_before)))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(torch.equal(parameter, before)
                            for parameter, before in zip(model.parameters(), model_before)))
        self.assertTrue(any(not torch.equal(parameter, before)
                            for parameter, before in zip(controller.parameters(), controller_before)))

    def test_saved_checkpoint_preserves_source_weights_and_adds_controller(self):
        import train_cfg

        source = {
            'args': {'dataset': 'cifar10', 'num_timesteps': 10},
            'model_state_dict': {'weight': torch.tensor([1.0])},
            'ema_state_dict': {'weight': torch.tensor([2.0])},
        }
        controller = AdaptiveCFG(10, hidden_size=4)
        optimizer = torch.optim.AdamW(controller.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cfg.pth'
            train_cfg.save_controller_checkpoint(
                path, source, controller, optimizer,
                {'adaptive_cfg_hidden_size': 4, 'adaptive_cfg_min_scale': 1.0,
                 'adaptive_cfg_max_scale': 8.0},
            )
            saved = torch.load(path, map_location='cpu', weights_only=False)

        self.assertEqual(saved['model_state_dict'], source['model_state_dict'])
        self.assertEqual(saved['ema_state_dict'], source['ema_state_dict'])
        self.assertIn('adaptive_cfg_state_dict', saved)
        self.assertIn('adaptive_cfg_optimizer_state_dict', saved)
        self.assertEqual(saved['args']['adaptive_cfg_hidden_size'], 4)
        self.assertEqual(source, {
            'args': {'dataset': 'cifar10', 'num_timesteps': 10},
            'model_state_dict': {'weight': torch.tensor([1.0])},
            'ema_state_dict': {'weight': torch.tensor([2.0])},
        })


    def test_save_every_rejects_nonpositive_intervals(self):
        import contextlib
        import io
        import train_cfg

        for interval in ('0', '-1'):
            with self.subTest(interval=interval), patch.object(sys, 'argv', [
                    'train_cfg.py', '--checkpoint', 'base.pth', '--output', 'cfg.pth',
                    '--save_every', interval]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    train_cfg.parse_args()
                self.assertEqual(caught.exception.code, 2)