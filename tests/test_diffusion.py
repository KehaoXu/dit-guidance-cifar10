import unittest
from unittest.mock import patch

import torch

from src.adaptive_cfg import AdaptiveCFG
from src.diffusion import GaussianDiffusion


class FixedEpsModel(torch.nn.Module):
    num_classes = 2

    def forward(self, x_t, timestep, labels):
        del timestep
        eps = (labels != self.num_classes).to(dtype=x_t.dtype)
        return eps.view(-1, 1, 1, 1).expand_as(x_t)


class PatternEpsModel(torch.nn.Module):
    num_classes = 2

    def forward(self, x_t, timestep, labels):
        del timestep
        conditional = torch.tensor(
            [0.0, 2.0, 0.0, 2.0], device=x_t.device, dtype=x_t.dtype,
        ).reshape(1, 1, 1, 4)
        conditional = conditional.expand_as(x_t)
        mask = (labels != self.num_classes).reshape(-1, 1, 1, 1)
        return torch.where(mask, conditional, torch.zeros_like(x_t))


class TrainableFixedEpsModel(torch.nn.Module):
    num_classes = 2

    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x_t, timestep, labels):
        del timestep
        eps = (labels != self.num_classes).to(dtype=x_t.dtype)
        return eps.view(-1, 1, 1, 1).expand_as(x_t) + self.offset


class ConstantAdaptiveCFG(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, timestep):
        return torch.full(timestep.shape, self.scale, device=timestep.device)


class CFGPPSamplingTests(unittest.TestCase):
    def test_cfgpp_uses_guided_x_start_and_unconditional_noise(self):
        """Catch the bug where CFG++ does not re-noise with unconditional noise."""
        diffusion = GaussianDiffusion(num_timesteps=2)
        model = FixedEpsModel()
        labels = torch.tensor([1])
        initial_noise = torch.full((1, 1, 1, 1), 0.25)

        with patch.object(torch, "randn", return_value=initial_noise):
            result = diffusion.cfgpp_sample_loop(
                model, initial_noise.shape, labels, cfg_scale=3.0, num_sampling_steps=1
            )

        alpha_0 = diffusion.alphas_cumprod[0]
        x_start_guided = (initial_noise - torch.sqrt(1 - alpha_0) * 3.0) / torch.sqrt(alpha_0)
        expected = torch.sqrt(alpha_0) * x_start_guided

        torch.testing.assert_close(result, expected)


class CFGRescaleTests(unittest.TestCase):
    def test_model_eps_applies_rescale_after_cfg_combination(self):
        diffusion = GaussianDiffusion(num_timesteps=2)
        image = torch.zeros(1, 1, 1, 4)

        result = diffusion._model_eps(
            PatternEpsModel(), image, torch.tensor([1]), torch.tensor([1]),
            cfg_scale=2.0, cfg_rescale=1.0,
        )

        torch.testing.assert_close(
            result, torch.tensor([[[[0.0, 2.0, 0.0, 2.0]]]]),
        )

    def test_cfgpp_model_eps_rescales_only_the_guided_prediction(self):
        diffusion = GaussianDiffusion(num_timesteps=2)
        image = torch.zeros(1, 1, 1, 4)

        guided, unconditional = diffusion._model_eps_cfgpp(
            PatternEpsModel(), image, torch.tensor([1]), torch.tensor([1]),
            cfg_scale=2.0, cfg_rescale=1.0,
        )

        torch.testing.assert_close(
            guided, torch.tensor([[[[0.0, 2.0, 0.0, 2.0]]]]),
        )
        torch.testing.assert_close(unconditional, torch.zeros_like(image))

    def test_rescale_rejects_values_outside_unit_interval(self):
        prediction = torch.zeros(1, 1, 1, 2)

        for value in (-0.1, 1.1, float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GaussianDiffusion.rescale_cfg(prediction, prediction, value)

    def test_full_rescale_matches_conditional_prediction_standard_deviation(self):
        conditional = torch.tensor([[[[0.0, 2.0, 0.0, 2.0]]]])
        guided = torch.tensor([[[[0.0, 4.0, 0.0, 4.0]]]])

        result = GaussianDiffusion.rescale_cfg(guided, conditional, 1.0)

        torch.testing.assert_close(result, conditional)

    def test_partial_rescale_blends_with_original_cfg_prediction(self):
        conditional = torch.tensor([[[[0.0, 2.0, 0.0, 2.0]]]])
        guided = torch.tensor([[[[0.0, 4.0, 0.0, 4.0]]]])

        result = GaussianDiffusion.rescale_cfg(guided, conditional, 0.5)

        torch.testing.assert_close(result, torch.tensor([[[[0.0, 3.0, 0.0, 3.0]]]]))

    def test_zero_variance_guided_prediction_stays_finite_and_unchanged(self):
        conditional = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
        guided = torch.ones_like(conditional)

        result = GaussianDiffusion.rescale_cfg(guided, conditional, 1.0)

        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result, guided)


class OnlineAdaptiveCFGLossTests(unittest.TestCase):
    def test_loss_only_updates_controller_and_restores_model_mode(self):
        diffusion = GaussianDiffusion(num_timesteps=10)
        model = TrainableFixedEpsModel().train()
        controller = AdaptiveCFG(10, hidden_size=8, min_scale=0.0,
                                 max_scale=4.0, initial_scale=1.0)
        images = torch.zeros(2, 1, 1, 1)
        labels = torch.tensor([0, 1])

        total, reconstruction, classification = diffusion.adaptive_cfg_loss(
            model, controller, images, labels, classification_weight=0.0,
            timestep=torch.tensor([2, 7]), noise=torch.full_like(images, 2.0),
        )
        total.backward()

        self.assertTrue(model.training)
        self.assertIsNone(model.offset.grad)
        self.assertTrue(any(parameter.grad is not None
                            for parameter in controller.parameters()))
        self.assertEqual((total.ndim, reconstruction.ndim, classification.ndim), (0, 0, 0))


class AdaptiveCFGSamplingTests(unittest.TestCase):
    def test_model_eps_uses_controller_scale_instead_of_fixed_scale(self):
        diffusion = GaussianDiffusion(num_timesteps=2)
        model = FixedEpsModel()
        image = torch.zeros(1, 1, 1, 1)

        result = diffusion._model_eps(
            model, image, torch.tensor([1]), torch.tensor([1]), cfg_scale=1.0,
            adaptive_cfg=ConstantAdaptiveCFG(3.0),
        )

        torch.testing.assert_close(result, torch.full_like(image, 3.0))


class CallRecordingEpsModel(torch.nn.Module):
    """FixedEpsModel that records the batch size of every forward pass."""

    num_classes = 2

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, x_t, timestep, labels):
        del timestep
        self.batch_sizes.append(x_t.shape[0])
        eps = (labels != self.num_classes).to(dtype=x_t.dtype)
        return eps.view(-1, 1, 1, 1).expand_as(x_t)


class GuidanceIntervalTests(unittest.TestCase):
    def _diffusion_and_controller(self, scale=3.0):
        from src.adaptive_cfg import IntervalCFG

        diffusion = GaussianDiffusion(num_timesteps=1000)
        sigma = ((1 - diffusion.alphas_cumprod) / diffusion.alphas_cumprod).sqrt()
        # t=200 is inside, t=800 is far outside for the linear schedule.
        controller = IntervalCFG(diffusion.alphas_cumprod, scale=scale,
                                 sigma_lo=float(sigma[100]), sigma_hi=float(sigma[300]))
        return diffusion, controller

    def test_unconditional_pass_is_skipped_outside_the_interval(self):
        diffusion, controller = self._diffusion_and_controller()
        model = CallRecordingEpsModel()
        x_t = torch.zeros(4, 1, 1, 1)
        labels = torch.ones(4, dtype=torch.long)

        eps = diffusion._model_eps(
            model, x_t, torch.full((4,), 800, dtype=torch.long), labels,
            cfg_scale=1.0, adaptive_cfg=controller,
        )

        self.assertEqual(model.batch_sizes, [4])
        torch.testing.assert_close(eps, torch.ones_like(x_t))

    def test_both_passes_run_inside_the_interval(self):
        diffusion, controller = self._diffusion_and_controller()
        model = CallRecordingEpsModel()
        x_t = torch.zeros(4, 1, 1, 1)
        labels = torch.ones(4, dtype=torch.long)

        eps = diffusion._model_eps(
            model, x_t, torch.full((4,), 200, dtype=torch.long), labels,
            cfg_scale=1.0, adaptive_cfg=controller,
        )

        self.assertEqual(model.batch_sizes, [8])
        # eps_cond = 1, eps_uncond = 0, so the guided prediction is the scale.
        torch.testing.assert_close(eps, torch.full_like(x_t, 3.0))