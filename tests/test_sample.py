import importlib
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch


def _save_image(_image, path):
    Path(path).write_bytes(b"png")


class SampleExportTests(unittest.TestCase):
    def test_parse_args_accepts_cfg_rescale(self):
        torch_module = types.ModuleType('torch')
        torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch_module.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False),
        )
        torchvision = types.ModuleType('torchvision')
        torchvision_utils = types.ModuleType('torchvision.utils')
        torchvision_utils.save_image = _save_image
        torchvision.utils = torchvision_utils
        dependencies = {
            'torch': torch_module,
            'torchvision': torchvision,
            'torchvision.utils': torchvision_utils,
            'src.utils': types.ModuleType('src.utils'),
            'src.diffusion': types.ModuleType('src.diffusion'),
            'src.model': types.ModuleType('src.model'),
        }
        dependencies['src.utils'].num_classes_from_state_dict = object()
        dependencies['src.diffusion'].GaussianDiffusion = object()
        dependencies['src.model'].DiT = object()

        with patch.dict(sys.modules, dependencies), patch.object(
            sys, 'argv',
            ['sample.py', '--checkpoint', 'model.pth', '--cfg_rescale', '0.7'],
        ):
            sys.modules.pop('sample', None)
            sample = importlib.import_module('sample')
            args = sample.parse_args()

        self.assertEqual(args.cfg_rescale, 0.7)

    def test_parse_args_rejects_cfg_rescale_outside_unit_interval(self):
        sys.modules.pop('sample', None)
        sample = importlib.import_module('sample')
        self.addCleanup(sys.modules.pop, 'sample', None)

        for value in ('-0.1', '1.1', 'nan'):
            stderr = io.StringIO()
            with self.subTest(value=value), patch.object(
                sys, 'argv',
                ['sample.py', '--checkpoint', 'model.pth', '--cfg_rescale', value],
            ), redirect_stderr(stderr), self.assertRaises(SystemExit):
                sample.parse_args()
            self.assertIn('cfg_rescale must lie in [0, 1]', stderr.getvalue())

    def test_parse_args_accepts_cfgpp_and_output_directory(self):
        """Catch a regression where the notebook's CFG++ command is rejected by argparse."""
        torch_module = types.ModuleType("torch")
        torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch_module.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        torchvision = types.ModuleType("torchvision")
        torchvision_utils = types.ModuleType("torchvision.utils")
        torchvision_utils.save_image = _save_image
        torchvision.utils = torchvision_utils
        dependencies = {
            "torch": torch_module,
            "torchvision": torchvision,
            "torchvision.utils": torchvision_utils,
            "src.utils": types.ModuleType("src.utils"),
            "src.diffusion": types.ModuleType("src.diffusion"),
            "src.model": types.ModuleType("src.model"),
        }
        dependencies["src.utils"].num_classes_from_state_dict = object()
        dependencies["src.diffusion"].GaussianDiffusion = object()
        dependencies["src.model"].DiT = object()

        with patch.dict(sys.modules, dependencies), patch.object(
            sys,
            "argv",
            ["sample.py", "--checkpoint", "model.pth", "--sampler", "cfgpp", "--output_dir", "generated"],
        ):
            sample = importlib.import_module("sample")
            args = sample.parse_args()

        self.assertEqual(args.sampler, "cfgpp")
        self.assertEqual(args.output_dir, "generated")

    def test_save_individual_images_writes_one_png_per_image(self):
        """Catch a regression where batch sampling only saves a preview grid."""
        torchvision = types.ModuleType("torchvision")
        torchvision_utils = types.ModuleType("torchvision.utils")
        torchvision_utils.save_image = _save_image
        torchvision.utils = torchvision_utils
        dependencies = {
            "torch": types.ModuleType("torch"),
            "torchvision": torchvision,
            "torchvision.utils": torchvision_utils,
            "src.utils": types.ModuleType("src.utils"),
            "src.diffusion": types.ModuleType("src.diffusion"),
            "src.model": types.ModuleType("src.model"),
        }
        dependencies["src.utils"].num_classes_from_state_dict = object()
        dependencies["src.diffusion"].GaussianDiffusion = object()
        dependencies["src.model"].DiT = object()

        with patch.dict(sys.modules, dependencies):
            sample = importlib.import_module("sample")
            with tempfile.TemporaryDirectory() as directory:
                written = sample.save_individual_images([object(), object(), object()], directory)
                self.assertEqual(
                    written,
                    [
                        Path(directory) / "000000.png",
                        Path(directory) / "000001.png",
                        Path(directory) / "000002.png",
                    ],
                )
                self.assertTrue(all(path.is_file() for path in written))

    def test_save_individual_images_includes_each_label_in_the_filename(self):
        """Generated files retain their corresponding class labels."""
        torchvision = types.ModuleType("torchvision")
        torchvision_utils = types.ModuleType("torchvision.utils")
        torchvision_utils.save_image = _save_image
        torchvision.utils = torchvision_utils
        dependencies = {
            "torch": types.ModuleType("torch"),
            "torchvision": torchvision,
            "torchvision.utils": torchvision_utils,
            "src.utils": types.ModuleType("src.utils"),
            "src.diffusion": types.ModuleType("src.diffusion"),
            "src.model": types.ModuleType("src.model"),
        }
        dependencies["tqdm"] = types.ModuleType("tqdm")
        dependencies["tqdm"].tqdm = lambda iterable: iterable
        dependencies["src.utils"].num_classes_from_state_dict = object()
        dependencies["src.diffusion"].GaussianDiffusion = object()
        dependencies["src.model"].DiT = object()

        with patch.dict(sys.modules, dependencies):
            sample = importlib.import_module("sample")
            with tempfile.TemporaryDirectory() as directory:
                written = sample.save_individual_images(
                    [object(), object(), object()], directory, labels=[3, 7, 1]
                )

        self.assertEqual(
            written,
            [
                Path(directory) / "000000_label_3.png",
                Path(directory) / "000001_label_7.png",
                Path(directory) / "000002_label_1.png",
            ],
        )


class SampleMainTests(unittest.TestCase):
    def test_preview_contains_every_requested_label_without_output_dir(self):
        """More --labels than --num_images must not be split and partly discarded."""
        from src.model import DiT
        sys.modules.pop('sample', None)
        sample = importlib.import_module('sample')
        self.addCleanup(sys.modules.pop, 'sample', None)
        model = DiT(image_size=4, patch_size=2, hidden_size=8, depth=1,
                    num_heads=2, num_classes=3)
        train_args = {'image_size': 4, 'patch_size': 2, 'hidden_size': 8, 'depth': 1,
                      'num_heads': 2, 'mlp_ratio': 4.0, 'class_dropout_prob': 0.1,
                      'num_timesteps': 10, 'beta_schedule': 'linear'}
        labels = [str(index % 3) for index in range(20)]
        saved = []

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pth'
            torch.save({'args': train_args, 'ema_state_dict': model.state_dict()}, checkpoint)
            argv = ['sample.py', '--checkpoint', str(checkpoint), '--device', 'cpu',
                    '--num_sampling_steps', '2', '--output', str(Path(directory) / 'grid.png'),
                    '--labels', *labels]
            with patch.object(sys, 'argv', argv), patch.object(
                    sample, 'save_image', lambda images, *args, **kwargs: saved.append(images)):
                sample.main()

        self.assertEqual([images.shape[0] for images in saved], [20])


class AdaptiveCFGSampleTests(unittest.TestCase):
    def test_loader_reconstructs_controller_from_checkpoint(self):
        from src.adaptive_cfg import AdaptiveCFG
        sys.modules.pop('sample', None)
        sample = importlib.import_module('sample')
        self.addCleanup(sys.modules.pop, 'sample', None)
        source = AdaptiveCFG(20, hidden_size=6, min_scale=0.5,
                             max_scale=5.0, initial_scale=2.0)
        with torch.no_grad():
            source.network[-1].bias.fill_(0.5)
        checkpoint = {
            'args': {'num_timesteps': 20, 'cfg_scale': 2.0,
                     'adaptive_cfg_hidden_size': 6,
                     'adaptive_cfg_min_scale': 0.5,
                     'adaptive_cfg_max_scale': 5.0},
            'adaptive_cfg_state_dict': source.state_dict(),
        }

        loaded = sample.load_adaptive_cfg(checkpoint, 'cpu', requested=True)

        torch.testing.assert_close(
            loaded(torch.tensor([0, 19])), source(torch.tensor([0, 19]))
        )
        self.assertFalse(loaded.training)

    def test_loader_falls_back_for_checkpoint_without_adaptive_state(self):
        sys.modules.pop('sample', None)
        sample = importlib.import_module('sample')
        self.addCleanup(sys.modules.pop, 'sample', None)
        checkpoint = {'args': {'num_timesteps': 20, 'cfg_scale': 4.0}}

        loaded = sample.load_adaptive_cfg(checkpoint, 'cpu', requested=True)

        self.assertIsNone(loaded)


class GuidanceIntervalSampleTests(unittest.TestCase):
    def _sample_module(self):
        sys.modules.pop('sample', None)
        module = importlib.import_module('sample')
        self.addCleanup(sys.modules.pop, 'sample', None)
        return module

    def test_parse_args_accepts_a_guidance_interval(self):
        sample = self._sample_module()

        with patch.object(sys, 'argv', [
            'sample.py', '--checkpoint', 'model.pth',
            '--guidance_interval', '0.19', '1.61',
        ]):
            args = sample.parse_args()

        self.assertEqual(args.guidance_interval, [0.19, 1.61])

    def test_builder_returns_a_controller_covering_the_requested_interval(self):
        from src.diffusion import GaussianDiffusion
        sample = self._sample_module()
        diffusion = GaussianDiffusion(num_timesteps=1000)

        controller = sample.build_guidance_interval(
            diffusion, cfg_scale=8.0, interval=[0.19, 1.61], device='cpu')

        scales = controller(torch.tensor([10, 200, 900]))
        torch.testing.assert_close(scales, torch.tensor([1.0, 8.0, 1.0]))

    def test_builder_returns_none_without_an_interval(self):
        from src.diffusion import GaussianDiffusion
        sample = self._sample_module()

        built = sample.build_guidance_interval(
            GaussianDiffusion(num_timesteps=10), cfg_scale=8.0,
            interval=None, device='cpu')

        self.assertIsNone(built)