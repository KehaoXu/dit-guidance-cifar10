import unittest

import torch

from src.adaptive_cfg import AdaptiveCFG, IntervalCFG


class AdaptiveCFGTests(unittest.TestCase):
    def test_initial_output_is_the_configured_scale_for_every_timestep(self):
        controller = AdaptiveCFG(
            num_timesteps=1000,
            hidden_size=8,
            min_scale=1.0,
            max_scale=8.0,
            initial_scale=4.0,
        )

        scales = controller(torch.tensor([0, 17, 500, 999]))

        torch.testing.assert_close(scales, torch.full((4,), 4.0))

    def test_outputs_remain_bounded_and_parameters_are_differentiable(self):
        controller = AdaptiveCFG(10, hidden_size=8, min_scale=0.5,
                                 max_scale=3.0, initial_scale=1.0)
        with torch.no_grad():
            controller.network[-1].weight.fill_(100.0)

        scales = controller(torch.arange(10))
        self.assertTrue(torch.all(scales >= 0.5))
        self.assertTrue(torch.all(scales <= 3.0))
        scales.sum().backward()
        self.assertTrue(any(parameter.grad is not None
                            for parameter in controller.parameters()))

    def test_invalid_configuration_is_rejected(self):
        invalid = [
            {'num_timesteps': 1},
            {'num_timesteps': 10, 'hidden_size': 0},
            {'num_timesteps': 10, 'min_scale': 2.0, 'max_scale': 2.0},
            {'num_timesteps': 10, 'min_scale': 1.0, 'max_scale': 2.0,
             'initial_scale': 3.0},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AdaptiveCFG(**kwargs)


class IntervalCFGTests(unittest.TestCase):
    @staticmethod
    def _alphas_cumprod_for(sigmas):
        """Build alphas_cumprod whose noise levels are exactly `sigmas`."""
        sigma = torch.tensor(sigmas, dtype=torch.float32)
        return 1.0 / (1.0 + sigma ** 2)

    def test_scale_is_applied_inside_the_interval_and_one_outside(self):
        alphas_cumprod = self._alphas_cumprod_for([0.1, 0.5, 1.0, 2.0, 4.0])

        controller = IntervalCFG(alphas_cumprod, scale=8.0,
                                 sigma_lo=0.5, sigma_hi=2.0)
        scales = controller(torch.arange(5))

        torch.testing.assert_close(scales, torch.tensor([1.0, 8.0, 8.0, 8.0, 1.0]))

    def test_outside_the_interval_the_scale_is_one_not_zero(self):
        """A scale of 0 would drop the class label entirely at high noise."""
        alphas_cumprod = self._alphas_cumprod_for([0.1, 1.0])

        controller = IntervalCFG(alphas_cumprod, scale=8.0,
                                 sigma_lo=0.5, sigma_hi=2.0)

        self.assertEqual(controller(torch.tensor([0])).item(), 1.0)

    def test_invalid_configuration_is_rejected(self):
        alphas_cumprod = self._alphas_cumprod_for([0.1, 1.0, 4.0])
        invalid = [
            {'scale': 8.0, 'sigma_lo': 2.0, 'sigma_hi': 0.5},
            {'scale': 8.0, 'sigma_lo': 1.0, 'sigma_hi': 1.0},
            {'scale': 8.0, 'sigma_lo': -1.0, 'sigma_hi': 1.0},
            {'scale': float('nan'), 'sigma_lo': 0.5, 'sigma_hi': 2.0},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                IntervalCFG(alphas_cumprod, **kwargs)

    def test_buffer_moves_with_the_module(self):
        alphas_cumprod = self._alphas_cumprod_for([0.1, 1.0, 4.0])

        controller = IntervalCFG(alphas_cumprod, scale=8.0,
                                 sigma_lo=0.5, sigma_hi=2.0).to(torch.float64)

        self.assertEqual(controller(torch.tensor([1])).dtype, torch.float64)