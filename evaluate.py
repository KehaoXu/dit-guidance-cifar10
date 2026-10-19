from pathlib import Path

import argparse
import re
import shutil
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR-10 samples with FID, cFID, IS and ResNet20 class accuracy."
    )
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--real_dir", type=str, default="./data")
    parser.add_argument(
        "--local_cache_dir",
        type=str,
        default=None,
        help="Optional local directory to copy generated and real datasets into before evaluation.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum generated images to evaluate per sampler (default: all images).",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=None,
        help="Optional experiment directory names to evaluate (default: discover all).",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="evaluation_results.csv")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate experiments that already have a row in --output instead of skipping them.",
    )
    return parser.parse_args()


class ImagePathDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = read_image(str(self.paths[index]))
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"Expected RGB image with shape [3, H, W], got {image.shape}"
            )
        return image


def generated_loader(paths, batch_size, num_workers, device):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.startswith("cuda"),
    }
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=4,
        )
    return DataLoader(ImagePathDataset(paths), **loader_kwargs)


def copy_directory_to_cache(source_dir, cache_root, cache_name):
    source_dir = Path(source_dir)
    destination = Path(cache_root) / cache_name

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if destination.resolve() == source_dir.resolve():
        return source_dir
    if destination.resolve().is_relative_to(source_dir.resolve()):
        raise ValueError("The local cache directory must not be inside the source directory")

    files = [path for path in source_dir.rglob("*") if path.is_file()]
    destination.mkdir(parents=True, exist_ok=True)
    for source_file in tqdm(files, desc=f"Copying {cache_name}", unit="file"):
        destination_file = destination / source_file.relative_to(source_dir)
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)
    return destination


def prepare_evaluation_directories(generated_dir, real_dir, local_cache_dir):
    if local_cache_dir is None:
        return generated_dir, real_dir
    cache_root = Path(local_cache_dir)
    return (
        copy_directory_to_cache(generated_dir, cache_root, "generated"),
        copy_directory_to_cache(real_dir, cache_root, "real"),
    )


def generated_image_paths(experiment_dir, max_images=None):
    generated_paths = sorted(experiment_dir.glob("batch_*/*.png"))
    if max_images is not None:
        return generated_paths[:max_images]
    return generated_paths


def parse_cfg_scale(encoded_scale):
    if "." in encoded_scale:
        return float(encoded_scale)
    if len(encoded_scale) == 1:
        return float(encoded_scale)
    return int(encoded_scale) / 10


def parse_experiment_metadata(experiment_name):
    steps_match = re.search(r"(?:^|[_-])steps?(\d+)(?:[_-]|$)", experiment_name)
    scale_match = re.search(r"(?:^|[_-])scale(\d+(?:\.\d+)?)(?:[_-]|$)", experiment_name)
    if scale_match is None:
        scale_match = re.search(r"(?:^|[_-])cfg(?:pp)?(\d+(?:\.\d+)?)(?:[_-]|$)", experiment_name)
    # For guidance-interval runs cfg_scale is the strength applied inside the
    # interval; the bounds themselves stay in the sampler name.
    return {
        "cfg_scale": None if scale_match is None else parse_cfg_scale(scale_match.group(1)),
        "sampling_steps": None if steps_match is None else int(steps_match.group(1)),
    }


def parse_generated_label(image_path):
    label_match = re.search(r"_label_(\d+)\.png$", Path(image_path).name)
    if label_match is None:
        raise ValueError(f"Could not parse a class label from generated image path: {image_path}")
    return int(label_match.group(1))


def discover_experiment_dirs(generated_dir, samplers=None, skip=()):
    generated_dir = Path(generated_dir)
    if not generated_dir.is_dir():
        raise FileNotFoundError(f"Generated samples directory does not exist: {generated_dir}")
    if samplers is not None:
        experiment_dirs = [(sampler, generated_dir / sampler) for sampler in samplers]
    else:
        experiment_dirs = [(path.name, path) for path in sorted(generated_dir.iterdir()) if path.is_dir()]
    # Drop skipped names before globbing: listing every batch is slow on
    # network-mounted storage, and a skipped experiment never needs the listing.
    experiment_dirs = [(name, path) for name, path in experiment_dirs if name not in skip]
    experiment_dirs = [(name, path) for name, path in experiment_dirs if generated_image_paths(path)]
    if not experiment_dirs and not skip:
        raise FileNotFoundError(f"No generated PNG files found under experiment directories in {generated_dir}")
    return experiment_dirs


def create_cifar10_classifier(device):
    """Load frozen CIFAR-10 ResNet20; Torch Hub caches the downloaded weights."""
    classifier = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_resnet20",
        pretrained=True,
        trust_repo=True,
    ).to(device).eval()
    classifier.requires_grad_(False)
    return classifier


def predict_generated_labels(images, classifier):
    """Classify uint8 RGB PNG batches using the pretrained model's normalization."""
    images = images.float() / 255.0
    # https://github.com/chenyaofo/image-classification-codebase/blob/master/conf/cifar10.conf
    mean = images.new_tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = images.new_tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    return classifier((images - mean) / std).argmax(dim=1)


def create_fid(device):
    return FrechetInceptionDistance(feature=2048, normalize=False, reset_real_features=False).to(device)


def create_class_fids(device, num_classes=10):
    return [create_fid(device) for _ in range(num_classes)]


def update_class_fids(images, labels, class_fids, real):
    for class_id, class_fid in enumerate(class_fids):
        class_mask = labels == class_id
        if class_mask.any():
            class_fid.update(images[class_mask], real=real)


def progress_batches(batches, description):
    return tqdm(batches, desc=description, unit="batch", leave=False)


@torch.no_grad()
def evaluate_sampler(
    sampler,
    generated_dir,
    fid,
    class_fids,
    batch_size,
    num_workers,
    device,
    max_images=None,
    classifier=None,
):
    sampler_dir = generated_dir / sampler

    generated_paths = generated_image_paths(sampler_dir, max_images=max_images)

    if not generated_paths:
        raise FileNotFoundError(
            f"No PNG files found under {sampler_dir}"
        )

    if classifier is None:
        classifier = create_cifar10_classifier(device)

    inception = InceptionScore(
        splits=10,
        normalize=False,
    ).to(device)

    gen_loader = generated_loader(
        generated_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    generated_labels = [parse_generated_label(path) for path in generated_paths]
    image_offset = 0
    correct_predictions = 0
    for fake_images in progress_batches(gen_loader, f"Evaluating {sampler}"):
        fake_images = fake_images.to(
            device,
            non_blocking=device.startswith("cuda"),
        )
        fid.update(fake_images, real=False)
        batch_image_count = len(fake_images)
        batch_labels = torch.tensor(
            generated_labels[image_offset:image_offset + batch_image_count],
            device=device,
        )
        update_class_fids(fake_images, batch_labels, class_fids, real=False)
        predicted_labels = predict_generated_labels(fake_images, classifier)
        correct_predictions += (predicted_labels == batch_labels).sum().item()
        image_offset += batch_image_count
        inception.update(fake_images)

    inception_mean, inception_std = inception.compute()
    fid_score = float(fid.compute())
    cfid_score = sum(float(class_fid.compute()) for class_fid in class_fids) / len(class_fids)
    fid.reset()
    for class_fid in class_fids:
        class_fid.reset()

    metadata = parse_experiment_metadata(sampler)

    return {
        "sampler": sampler,
        **metadata,
        "num_images": len(generated_paths),
        "class_accuracy": correct_predictions / len(generated_paths),
        "cfid": cfid_score,
        "fid": fid_score,
        "inception_score_mean": float(inception_mean),
        "inception_score_std": float(inception_std),
    }

def evaluate_samplers(
    samplers,
    generated_dir,
    fid,
    class_fids,
    batch_size,
    num_workers,
    device,
    max_images=None,
    on_result=None,
    classifier=None,
):
    if classifier is None:
        classifier = create_cifar10_classifier(device)
    results = []
    for sampler in samplers:
        result = evaluate_sampler(
            sampler=sampler,
            generated_dir=generated_dir,
            fid=fid,
            class_fids=class_fids,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            max_images=max_images,
            classifier=classifier,
        )
        results.append(result)
        if on_result is not None:
            on_result(results)
    return results


def load_existing_results(output):
    """Read previously written results, or an empty frame on the first run."""
    output_path = Path(output)
    if not output_path.exists():
        return pd.DataFrame()
    return pd.read_csv(output_path)


def evaluated_experiments(existing_results, overwrite=False):
    """Names the results file already covers, which discovery skips outright."""
    if overwrite or existing_results.empty or "sampler" not in existing_results:
        return set()
    return set(existing_results["sampler"])


def merge_results(sampler_results, existing_results=None):
    """Append this run's rows after the rows the results file already held."""
    frame = pd.DataFrame(sampler_results)
    if existing_results is not None and not existing_results.empty:
        frame = pd.concat([existing_results, frame], ignore_index=True)
    return frame


def write_evaluation_results(sampler_results, output, existing_results=None):
    merge_results(sampler_results, existing_results).to_csv(output, index=False)


def main():
    args = parse_args()
    generated_dir, real_dir = prepare_evaluation_directories(
        Path(args.generated_dir),
        Path(args.real_dir),
        args.local_cache_dir,
    )

    # Resolve what still needs evaluating before paying for the real-image pass.
    existing_results = load_existing_results(args.output)
    skipped = evaluated_experiments(existing_results, args.overwrite)
    if skipped:
        print(f"Reusing {len(skipped)} results already in {args.output}", flush=True)
    print(f"Discovering experiments under {generated_dir}", flush=True)
    experiments = discover_experiment_dirs(generated_dir, args.samplers, skip=skipped)
    samplers = [name for name, _ in experiments]
    if not samplers:
        print(f"Nothing new to evaluate; {args.output} already covers every discovered experiment.",
              flush=True)
        return
    print(f"Evaluating {len(samplers)} experiments: {', '.join(samplers)}", flush=True)
    if not existing_results.empty and "sampler" in existing_results:
        existing_results = existing_results[~existing_results["sampler"].isin(samplers)]

    real_dataset = datasets.CIFAR10(
        root=real_dir,
        train=False,
        download=True,
        transform=transforms.PILToTensor(),
    )

    real_loader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    fid = create_fid(args.device)
    class_fids = create_class_fids(args.device)

    for real_images, real_labels in tqdm(
        real_loader,
        desc="Extracting real FID features",
        unit="batch",
    ):
        real_images = real_images.to(
            args.device,
            non_blocking=args.device.startswith("cuda"),
        )
        fid.update(real_images, real=True)
        update_class_fids(real_images, real_labels.to(args.device), class_fids, real=True)

    sampler_results = evaluate_samplers(
        samplers,
        generated_dir=generated_dir,
        fid=fid,
        class_fids=class_fids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_images=args.max_images,
        on_result=lambda results: write_evaluation_results(
            results, args.output, existing_results
        ),
    )
    print(merge_results(sampler_results, existing_results).to_string(index=False))


if __name__ == "__main__":
    main()
