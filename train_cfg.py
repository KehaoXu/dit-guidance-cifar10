import argparse
import copy
import math
from pathlib import Path

import matplotlib
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.adaptive_cfg import AdaptiveCFG
from src.dataset import prepare_dataset
from src.diffusion import GaussianDiffusion
from src.model import DiT
from src.utils import num_classes_from_state_dict

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def default_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train an adaptive CFG controller against a frozen EMA DiT checkpoint.'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='DiT checkpoint whose EMA weights remain frozen.')
    parser.add_argument('--output', required=True,
                        help='Output checkpoint containing the trained controller.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint, CFG curve and preview every N epochs and at the final epoch.')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--imagenet_path', default=None)
    parser.add_argument('--adaptive_cfg_lr', type=float, default=1e-3)
    parser.add_argument('--adaptive_cfg_hidden_size', type=int, default=32)
    parser.add_argument('--adaptive_cfg_min_scale', type=float, default=1.0)
    parser.add_argument('--adaptive_cfg_max_scale', type=float, default=8.0)
    parser.add_argument('--adaptive_cfg_classification_weight', type=float, default=0.01,
                        help='Weight of CIFAR-10 classification CE; 0 disables the classifier.')
    parser.add_argument('--adaptive_cfg_classification_min_snr', type=float, default=1.0,
                        help='Apply classification CE only where signal-to-noise ratio >= this value.')
    parser.add_argument('--initial_scale', type=float, default=None)
    parser.add_argument('--preview_sampler', default='ddim',
                        choices=['ddpm', 'ddim', 'cfgpp'])
    parser.add_argument('--preview_num_sampling_steps', type=int, default=50)
    parser.add_argument('--preview_num_images', type=int, default=10)
    parser.add_argument('--preview_seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=default_device())
    args = parser.parse_args()
    if args.save_every < 1:
        parser.error('--save_every must be a positive integer.')
    for name in ('adaptive_cfg_classification_weight', 'adaptive_cfg_classification_min_snr'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f'--{name} must be finite and non-negative.')
    return args


def checkpoint_args(checkpoint):
    saved_args = checkpoint.get('args', {})
    if isinstance(saved_args, dict):
        return dict(saved_args)
    return vars(saved_args)


def num_classes_from_args(saved_args):
    if 'num_classes' in saved_args:
        return saved_args['num_classes']
    return 1000 if saved_args.get('dataset') == 'imagenet' else 10


def build_frozen_ema_model(checkpoint, device):
    saved_args = checkpoint_args(checkpoint)
    model = DiT(
        image_size=saved_args.get('image_size', 32),
        image_channel=saved_args.get('image_channel', 3),
        patch_size=saved_args.get('patch_size', 2),
        hidden_size=saved_args.get('hidden_size', 384),
        depth=saved_args.get('depth', 12),
        num_heads=saved_args.get('num_heads', 6),
        mlp_ratio=saved_args.get('mlp_ratio', 4.0),
        num_classes=num_classes_from_state_dict(checkpoint['ema_state_dict']),
        class_dropout_prob=saved_args.get('class_dropout_prob', 0.1),
    ).to(device)
    model.load_state_dict(checkpoint['ema_state_dict'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class CFGClassifier(torch.nn.Module):
    '''Adapt predicted diffusion x0 to frozen CIFAR-10 classifier logits.'''

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier.requires_grad_(False).eval()
        # Match evaluate.py and the pretrained ResNet20 training normalization.
        self.register_buffer('mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1))
        self.eval()

    def forward(self, images):
        # Keep preprocessing differentiable; clipping saturates out-of-range pixels.
        images = (images.clamp(-1, 1) + 1) / 2
        return self.classifier((images - self.mean) / self.std)


def create_cfg_classifier(saved_args, device):
    if (saved_args.get('dataset', 'cifar10') != 'cifar10'
            or num_classes_from_args(saved_args) != 10
            or saved_args.get('image_channel', 3) != 3
            or saved_args.get('image_size', 32) != 32):
        raise ValueError(
            'Classification loss supports 32x32 RGB CIFAR-10 checkpoints only; '
            'set --adaptive_cfg_classification_weight 0 for noise-only training.'
        )
    # Same pretrained model as evaluate.py, without importing its metric dependencies.
    classifier = torch.hub.load(
        'chenyaofo/pytorch-cifar-models', 'cifar10_resnet20',
        pretrained=True, trust_repo=True,
    )
    return CFGClassifier(classifier).to(device).eval()


def train_controller_step(diffusion, frozen_model, controller, optimizer,
                          images, labels, classifier=None, classification_weight=0.01,
                          classification_min_snr=1.0):
    optimizer.zero_grad(set_to_none=True)
    total, reconstruction, classification = diffusion.adaptive_cfg_loss(
        frozen_model, controller, images, labels, classifier=classifier,
        classification_weight=classification_weight,
        classification_min_snr=classification_min_snr,
    )
    total.backward()
    optimizer.step()
    return total.detach(), reconstruction, classification


def save_controller_checkpoint(path, source_checkpoint, controller, optimizer,
                               controller_args):
    checkpoint = copy.deepcopy(source_checkpoint)
    saved_args = checkpoint_args(checkpoint)
    saved_args.pop('adaptive_cfg_smoothness', None)
    saved_args.update(controller_args)
    checkpoint['args'] = saved_args
    checkpoint['adaptive_cfg_state_dict'] = controller.state_dict()
    checkpoint['adaptive_cfg_optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path)


@torch.no_grad()
def save_epoch_preview(diffusion, frozen_model, controller, num_classes, image_size,
                       args, epoch, output_dir):
    output_path = Path(output_dir) / 'cfg_previews' / f'epoch_{epoch:03d}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_images = min(num_classes, args.preview_num_images)
    labels = torch.arange(num_images, device=args.device)
    device = torch.device(args.device)
    cuda_devices = list(range(torch.cuda.device_count())) if device.type == 'cuda' else []
    was_training = controller.training
    controller.eval()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(args.preview_seed)
            images = diffusion.sample(
                frozen_model, num_images, labels, image_size, image_channel=3,
                sampler=args.preview_sampler,
                num_sampling_steps=args.preview_num_sampling_steps,
                cfg_scale=1.0, device=args.device, adaptive_cfg=controller,
            )
    finally:
        controller.train(was_training)
    save_image((images.clamp(-1, 1) + 1) / 2, output_path, nrow=num_images)
    return output_path


@torch.no_grad()
def save_cfg_curve(controller, num_timesteps, epoch, output_dir, device):
    output_path = Path(output_dir) / 'cfg_curves' / f'epoch_{epoch:03d}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    was_training = controller.training
    controller.eval()
    try:
        timesteps = torch.arange(num_timesteps, device=device)
        scales = controller(timesteps).cpu()
    finally:
        controller.train(was_training)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(timesteps.cpu(), scales)
    axis.set_xlabel('Timestep')
    axis.set_ylabel('CFG scale')
    axis.set_ylim(controller.min_scale, controller.max_scale)
    axis.set_title(f'Adaptive CFG after epoch {epoch}')
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    source_checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved_args = checkpoint_args(source_checkpoint)
    classifier = (create_cfg_classifier(saved_args, args.device)
                  if args.adaptive_cfg_classification_weight > 0 else None)
    frozen_model = build_frozen_ema_model(source_checkpoint, args.device)

    batch_size = args.batch_size or saved_args.get('batch_size', 128)
    num_workers = args.num_workers if args.num_workers is not None else saved_args.get('num_workers', 4)
    data_root = args.data_root or saved_args.get('data_root', './data')
    imagenet_path = args.imagenet_path or saved_args.get('imagenet_path', './ImagenetHighResolution')
    train_loader, num_classes = prepare_dataset(
        saved_args.get('dataset', 'cifar10'), data_root, imagenet_path,
        saved_args.get('image_size', 32), batch_size, num_workers,
        min_image_size=saved_args.get('min_image_size', 0),
    )
    if num_classes != frozen_model.num_classes:
        raise ValueError('The checkpoint and training dataset have different class counts.')

    initial_scale = args.initial_scale
    if initial_scale is None:
        initial_scale = saved_args.get('cfg_scale', 4.0)
    controller = AdaptiveCFG(
        saved_args.get('num_timesteps', 1000),
        hidden_size=args.adaptive_cfg_hidden_size,
        min_scale=args.adaptive_cfg_min_scale,
        max_scale=args.adaptive_cfg_max_scale,
        initial_scale=initial_scale,
    ).to(args.device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.adaptive_cfg_lr)
    diffusion = GaussianDiffusion(
        saved_args.get('num_timesteps', 1000), saved_args.get('beta_schedule', 'linear')
    ).to(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    controller_args = {
        'adaptive_cfg_hidden_size': args.adaptive_cfg_hidden_size,
        'adaptive_cfg_min_scale': args.adaptive_cfg_min_scale,
        'adaptive_cfg_max_scale': args.adaptive_cfg_max_scale,
        'adaptive_cfg_classification_weight': args.adaptive_cfg_classification_weight,
        'adaptive_cfg_classification_min_snr': args.adaptive_cfg_classification_min_snr,
        'adaptive_cfg_save_every': args.save_every,
    }

    for epoch in range(args.epochs):
        progress = tqdm(train_loader, desc=f'CFG epoch {epoch + 1}/{args.epochs}')
        for images, labels in progress:
            images = images.to(args.device)
            labels = labels.to(args.device)
            total, reconstruction, classification = train_controller_step(
                diffusion, frozen_model, controller, optimizer, images, labels,
                classifier=classifier,
                classification_weight=args.adaptive_cfg_classification_weight,
                classification_min_snr=args.adaptive_cfg_classification_min_snr,
            )
            progress.set_postfix(
                loss=f'{total.item():.4f}', mse=f'{reconstruction.item():.4f}',
                ce=f'{classification.item():.4f}',
            )

        if (epoch + 1) % args.save_every != 0 and epoch + 1 != args.epochs:
            continue

        checkpoint_path = output_path.with_name(
            f'{output_path.stem}_epoch_{epoch + 1:03d}{output_path.suffix}'
        )
        save_controller_checkpoint(
            checkpoint_path, source_checkpoint, controller, optimizer, controller_args,
        )
        print(f'Saved adaptive CFG checkpoint -> {checkpoint_path}')
        preview_path = save_epoch_preview(
            diffusion, frozen_model, controller, num_classes,
            saved_args.get('image_size', 32), args, epoch + 1, output_path.parent,
        )
        curve_path = save_cfg_curve(
            controller, saved_args.get('num_timesteps', 1000), epoch + 1,
            output_path.parent, args.device,
        )
        print(f'Saved CFG preview -> {preview_path}')
        print(f'Saved CFG curve -> {curve_path}')

    save_controller_checkpoint(
        output_path, source_checkpoint, controller, optimizer,
        controller_args,
    )
    print(f'Saved adaptive CFG checkpoint -> {output_path}')


if __name__ == '__main__':
    main()