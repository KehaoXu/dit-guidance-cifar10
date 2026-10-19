import io
import unittest

import torch

from src.adaptive_cfg import AdaptiveCFG
from src.utils import EMA, load_checkpoint, save_checkpoint


def training_objects():
    model = torch.nn.Linear(1, 1)
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=10
    )
    return model, ema, optimizer, scheduler


class AdaptiveCFGCheckpointTests(unittest.TestCase):
    def test_round_trip_restores_controller_and_optimizer(self):
        model, ema, optimizer, scheduler = training_objects()
        controller = AdaptiveCFG(10, hidden_size=4)
        controller_optimizer = torch.optim.AdamW(controller.parameters(), lr=1e-3)
        controller(torch.tensor([1])).sum().backward()
        controller_optimizer.step()
        expected = controller(torch.tensor([5])).detach().clone()

        stream = io.BytesIO()
        save_checkpoint(
            stream, model, ema, optimizer, scheduler, 2, 7, {},
            adaptive_cfg=controller,
            adaptive_cfg_optimizer=controller_optimizer,
        )
        restored = AdaptiveCFG(10, hidden_size=4)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        stream.seek(0)
        load_checkpoint(
            stream, model, ema, optimizer, scheduler,
            adaptive_cfg=restored,
            adaptive_cfg_optimizer=restored_optimizer,
        )

        torch.testing.assert_close(restored(torch.tensor([5])), expected)
        self.assertTrue(restored_optimizer.state)

    def test_legacy_checkpoint_leaves_controller_at_initial_scale(self):
        model, ema, optimizer, scheduler = training_objects()
        stream = io.BytesIO()
        save_checkpoint(stream, model, ema, optimizer, scheduler, 0, 0, {})
        controller = AdaptiveCFG(10, hidden_size=4, initial_scale=3.0)

        stream.seek(0)
        load_checkpoint(stream, model, adaptive_cfg=controller)

        torch.testing.assert_close(
            controller(torch.tensor([5])), torch.tensor([3.0])
        )