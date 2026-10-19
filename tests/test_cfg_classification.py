import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from src.diffusion import GaussianDiffusion


class ConditionalPrediction(nn.Module):
    num_classes = 2

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, images, timestep, labels):
        value = (labels == self.num_classes).to(images.dtype) + self.bias
        return value.view(-1, 1, 1, 1).expand_as(images)


class ScalarController(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, timestep):
        return self.scale.expand_as(timestep)


class MeanClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, images):
        value = images.mean(dim=(1, 2, 3)) * self.gain
        return torch.stack([value, -value], dim=1)


class CFGClassificationTests(unittest.TestCase):
    def setUp(self):
        self.diffusion = GaussianDiffusion(1000)
        self.model = ConditionalPrediction()
        self.controller = ScalarController()
        self.classifier = MeanClassifier().eval()

    def loss(self, timesteps, **kwargs):
        images = torch.zeros(len(timesteps), 1, 2, 2)
        return self.diffusion.adaptive_cfg_loss(
            self.model, self.controller, images,
            torch.zeros(len(timesteps), dtype=torch.long),
            timestep=torch.tensor(timesteps), noise=torch.zeros_like(images),
            **kwargs,
        )

    def test_classification_gradient_can_increase_cfg_when_noise_mse_is_stationary(self):
        total, mse, ce = self.loss([0], classifier=self.classifier, classification_weight=0.1)
        self.assertAlmostEqual(mse.item(), 0.0)
        self.assertAlmostEqual(ce.item(), math.log(2), places=6)
        self.assertAlmostEqual(total.item(), 0.1 * math.log(2), places=6)
        total.backward()
        self.assertLess(self.controller.scale.grad.item(), 0.0)
        self.assertTrue(torch.isfinite(self.controller.scale.grad))
        self.assertIsNone(self.model.bias.grad)
        self.assertIsNone(self.classifier.gain.grad)
        self.assertFalse(mse.requires_grad)
        self.assertFalse(ce.requires_grad)

    def test_high_noise_samples_are_masked_and_ce_is_averaged_over_full_batch(self):
        _, _, ce = self.loss([0, 999], classifier=self.classifier,
                            classification_weight=0.1, classification_min_snr=1.0)
        self.assertAlmostEqual(ce.item(), math.log(2) / 2, places=6)

    def test_all_high_noise_batch_skips_classifier_and_still_backpropagates(self):
        with patch.object(self.classifier, 'forward', side_effect=AssertionError('must skip')):
            total, mse, ce = self.loss([999], classifier=self.classifier, classification_weight=0.1)
        self.assertEqual(ce.item(), 0.0)
        torch.testing.assert_close(total.detach(), mse)
        total.backward()
        self.assertTrue(torch.isfinite(self.controller.scale.grad))

    def test_zero_weight_allows_noise_only_training_without_classifier(self):
        total, mse, ce = self.loss([0], classification_weight=0.0)
        self.assertEqual(ce.item(), 0.0)
        torch.testing.assert_close(total.detach(), mse)

    def test_enabled_classification_requires_a_classifier(self):
        with self.assertRaisesRegex(ValueError, 'classifier'):
            self.loss([0], classification_weight=0.1)

    def test_invalid_classification_settings_are_rejected(self):
        for name in ('classification_weight', 'classification_min_snr'):
            for value in (-1.0, float('nan'), float('inf')):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.loss([0], classifier=self.classifier, **{name: value})

    def test_cfg_classifier_normalization_preserves_input_gradient_and_freezes_weights(self):
        import train_cfg
        raw = nn.Sequential(nn.Flatten(), nn.Linear(12, 10, bias=False))
        with torch.no_grad():
            raw[1].weight.fill_(1.0)
        with patch('torch.hub.load', return_value=raw):
            classifier = train_cfg.create_cfg_classifier({'dataset': 'cifar10'}, 'cpu')
        images = torch.zeros(1, 3, 2, 2, requires_grad=True)
        logits = classifier(images)
        expected = 4 * ((0.5 - 0.4914) / 0.2023
                        + (0.5 - 0.4822) / 0.1994
                        + (0.5 - 0.4465) / 0.2010)
        torch.testing.assert_close(logits, torch.full((1, 10), expected))
        logits.sum().backward()
        self.assertGreater(images.grad.abs().sum().item(), 0.0)
        self.assertFalse(classifier.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in classifier.parameters()))

    def test_cifar_classifier_rejects_other_datasets_before_loading_weights(self):
        import train_cfg
        with patch('torch.hub.load', side_effect=AssertionError('must not download')):
            with self.assertRaisesRegex(ValueError, 'CIFAR-10'):
                train_cfg.create_cfg_classifier({'dataset': 'imagenet'}, 'cpu')


class CFGTrainingIntegrationTests(unittest.TestCase):
    def test_main_trains_previews_and_saves_new_loss_settings(self):
        import sys
        import tempfile
        import train_cfg
        from src.model import DiT

        source_args = {
            'dataset': 'cifar10', 'image_size': 32, 'patch_size': 16,
            'hidden_size': 8, 'depth': 1, 'num_heads': 2,
            'num_timesteps': 10, 'beta_schedule': 'cosine',
            'adaptive_cfg_smoothness': 0.25,
        }
        model = DiT(image_size=32, patch_size=16, hidden_size=8, depth=1,
                    num_heads=2, num_classes=10)
        source = {'args': source_args, 'ema_state_dict': model.state_dict()}
        images = torch.zeros(2, 3, 32, 32)
        labels = torch.tensor([0, 1])
        for weight, epochs, interval, saved_epochs in ((0.03, 11, None, (10, 11)),
                                                       (0.0, 3, 2, (2, 3))):
            with self.subTest(weight=weight), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / 'source.pth'
                output = Path(directory) / 'cfg.pth'
                torch.save(source, checkpoint)
                raw = nn.Sequential(nn.Conv2d(3, 10, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
                before = [p.detach().clone() for p in raw.parameters()]
                argv = ['train_cfg.py', '--checkpoint', str(checkpoint), '--output', str(output),
                        '--epochs', str(epochs), '--device', 'cpu', '--num_workers', '0',
                        '--preview_num_sampling_steps', '2', '--preview_num_images', '2',
                        '--adaptive_cfg_classification_weight', str(weight),
                        '--adaptive_cfg_classification_min_snr', '0']
                if interval is not None:
                    argv.extend(['--save_every', str(interval)])
                download = {} if weight else {'side_effect': AssertionError('must not download')}
                with (patch.object(sys, 'argv', argv),
                      patch.object(train_cfg, 'prepare_dataset', return_value=([(images, labels)], 10)),
                      patch('torch.hub.load', return_value=raw, **download)):
                    train_cfg.main()
                saved = torch.load(output, weights_only=False)
                self.assertEqual(saved['args']['adaptive_cfg_classification_weight'], weight)
                self.assertEqual(saved['args']['adaptive_cfg_classification_min_snr'], 0)
                self.assertNotIn('adaptive_cfg_smoothness', saved['args'])
                self.assertIn('adaptive_cfg_state_dict', saved)
                self.assertIn('adaptive_cfg_optimizer_state_dict', saved)
                self.assertTrue(all(torch.isfinite(v).all()
                                    for v in saved['adaptive_cfg_state_dict'].values()))
                for name, tensor in source['ema_state_dict'].items():
                    torch.testing.assert_close(saved['ema_state_dict'][name], tensor)
                self.assertEqual(source['args']['adaptive_cfg_smoothness'], 0.25)
                for folder in ('cfg_previews', 'cfg_curves'):
                    self.assertEqual(
                        sorted(p.name for p in (Path(directory) / folder).glob('*.png')),
                        [f'epoch_{epoch:03d}.png' for epoch in saved_epochs],
                    )
                snapshots = sorted(Path(directory).glob('cfg_epoch_*.pth'))
                self.assertEqual([p.name for p in snapshots],
                                 [f'cfg_epoch_{epoch:03d}.pth' for epoch in saved_epochs])
                last_snapshot = torch.load(snapshots[-1], weights_only=False)
                for name, tensor in saved['adaptive_cfg_state_dict'].items():
                    torch.testing.assert_close(last_snapshot['adaptive_cfg_state_dict'][name], tensor)
                self.assertIn('adaptive_cfg_optimizer_state_dict', last_snapshot)
                self.assertTrue(all(p.grad is None and torch.equal(p, old)
                                    for p, old in zip(raw.parameters(), before)))