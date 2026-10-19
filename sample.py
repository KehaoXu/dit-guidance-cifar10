import argparse
import math
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.diffusion import GaussianDiffusion
from src.model import DiT
from src.utils import num_classes_from_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from a trained DiT checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="generated.png")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional directory for individual PNGs, suitable for metrics.")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false",
                        help="Sample from the raw training weights instead of EMA (EMA is recommended).")

    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Number of images to generate and save per batch.")
    parser.add_argument("--labels", type=int, nargs="+", default=None,
                        help="Explicit class ids to generate. Overrides --num_images.")
    parser.add_argument("--nrow", type=int, default=8)

    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim", "cfgpp"])
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument('--cfg_rescale', type=float, default=0.0,
                        help='Blend CFG with a per-sample standard-deviation rescale in [0, 1].')
    parser.add_argument('--adaptive_cfg', action='store_true',
                        help='Use the learned timestep-adaptive CFG schedule when available.')
    parser.add_argument('--guidance_interval', type=float, nargs=2, default=None,
                        metavar=('SIGMA_LO', 'SIGMA_HI'),
                        help='Apply --cfg_scale only where the noise level sigma lies in '
                             '[SIGMA_LO, SIGMA_HI]; outside it the plain conditional model '
                             'is used (scale 1). Try 0.19 1.61.')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    if not math.isfinite(args.cfg_rescale) or not 0.0 <= args.cfg_rescale <= 1.0:
        parser.error('cfg_rescale must lie in [0, 1].')
    return args


def load_adaptive_cfg(checkpoint, device, requested):
    if not requested:
        return None
    state = checkpoint.get('adaptive_cfg_state_dict')
    if state is None:
        print('Checkpoint has no adaptive CFG state; falling back to --cfg_scale.')
        return None

    from src.adaptive_cfg import AdaptiveCFG

    train_args = checkpoint['args']
    controller = AdaptiveCFG(
        train_args['num_timesteps'],
        hidden_size=train_args.get('adaptive_cfg_hidden_size', 32),
        min_scale=train_args.get('adaptive_cfg_min_scale', 1.0),
        max_scale=train_args.get('adaptive_cfg_max_scale', 8.0),
        initial_scale=train_args.get('cfg_scale', 4.0),
    )
    controller.load_state_dict(state)
    return controller.to(device).eval()


def build_guidance_interval(diffusion, cfg_scale, interval, device):
    """Build a fixed-scale guidance-interval controller, or None if unrequested."""
    if interval is None:
        return None

    from src.adaptive_cfg import IntervalCFG

    sigma_lo, sigma_hi = interval
    controller = IntervalCFG(diffusion.alphas_cumprod, cfg_scale, sigma_lo, sigma_hi)
    return controller.to(device).eval()


def save_individual_images(images, output_dir, labels=None):
    if labels is not None and len(images) != len(labels):
        raise ValueError("images and labels must have the same length")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, image in enumerate(images):
        label = None if labels is None else labels[index]
        label_value = label.item() if hasattr(label, "item") else label
        filename = (
            f"{index:06d}.png"
            if label is None
            else f"{index:06d}_label_{label_value}.png"
        )
        path = output_path / filename
        save_image(image, path)
        paths.append(path)
    return paths


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    train_args = checkpoint["args"]
    state_key = "ema_state_dict" if args.use_ema else "model_state_dict"
    num_classes = num_classes_from_state_dict(checkpoint[state_key])

    model = DiT(
        image_size=train_args["image_size"],
        image_channel=3,
        patch_size=train_args["patch_size"],
        hidden_size=train_args["hidden_size"],
        depth=train_args["depth"],
        num_heads=train_args["num_heads"],
        mlp_ratio=train_args["mlp_ratio"],
        num_classes=num_classes,
        class_dropout_prob=train_args["class_dropout_prob"],
    ).to(args.device)

    model.load_state_dict(checkpoint[state_key])
    model.eval()
    adaptive_cfg = load_adaptive_cfg(checkpoint, args.device, args.adaptive_cfg)
    print(f"Loaded {state_key} from {args.checkpoint} (num_classes={num_classes})")

    diffusion = GaussianDiffusion(train_args["num_timesteps"], train_args["beta_schedule"]).to(args.device)

    if args.guidance_interval is not None:
        if adaptive_cfg is not None:
            raise ValueError("--adaptive_cfg and --guidance_interval are mutually exclusive.")
        adaptive_cfg = build_guidance_interval(
            diffusion, args.cfg_scale, args.guidance_interval, args.device)
        print(f"Guidance interval sigma in {tuple(args.guidance_interval)}: "
              f"scale {args.cfg_scale} on {adaptive_cfg.num_active_timesteps}"
              f"/{diffusion.num_timesteps} timesteps")

    if args.labels is not None:
        all_labels = torch.tensor(args.labels, device=args.device)
    else:
        all_labels = None

    total_images = len(all_labels) if all_labels is not None else args.num_images
    # Without --output_dir only the preview grid is saved, so sample it in one batch.
    batch_size = args.batch_size if args.output_dir else total_images
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total_saved = 0
    for batch_index in tqdm(range(math.ceil(total_images / batch_size))):
        start = batch_index * batch_size
        stop = min(start + batch_size, total_images)
        torch.manual_seed(args.seed + batch_index)
        labels = (all_labels[start:stop] if all_labels is not None
                  else torch.randint(0, num_classes, (stop - start,), device=args.device))

        images = diffusion.sample(
            model, labels.shape[0], labels, train_args["image_size"], image_channel=3,
            sampler=args.sampler, num_sampling_steps=args.num_sampling_steps,
            cfg_scale=args.cfg_scale, device=args.device, adaptive_cfg=adaptive_cfg,
            cfg_rescale=args.cfg_rescale,
        )
        images = (images.clamp(-1, 1) + 1) / 2

        if batch_index == 0:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            save_image(images, args.output, nrow=args.nrow)
            print(f"Saved preview ({labels.shape[0]} images) -> {args.output}")
        if args.output_dir:
            batch_dir = Path(args.output_dir) / f"batch_{batch_index:03d}"
            paths = save_individual_images(images, batch_dir, labels)
            total_saved += len(paths)

    if args.output_dir:
        print(f"Saved {total_saved} individual images -> {args.output_dir}")


if __name__ == "__main__":
    main()