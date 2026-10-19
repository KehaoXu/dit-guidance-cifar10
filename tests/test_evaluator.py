import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# Import pandas once, outside patch.dict(sys.modules): a first import inside the patch is
# dropped on exit, and a later re-import creates a second, incompatible copy of pandas.
import pandas


class RecordingDataLoader:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs


def import_evaluator():
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    torch.no_grad = lambda: lambda function: function

    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = RecordingDataLoader
    torch_utils_data.Dataset = object
    torch_utils.data = torch_utils_data
    torch.utils = torch_utils

    torchvision = types.ModuleType("torchvision")
    torchvision_datasets = types.ModuleType("torchvision.datasets")
    torchvision_transforms = types.ModuleType("torchvision.transforms")
    torchvision_io = types.ModuleType("torchvision.io")
    class FakeImage:
        ndim = 3
        shape = (3, 32, 32)

        def __init__(self, path):
            self.path = path

    torchvision_io.read_image = FakeImage
    torchvision.datasets = torchvision_datasets
    torchvision.transforms = torchvision_transforms

    torchmetrics = types.ModuleType("torchmetrics")
    torchmetrics_image = types.ModuleType("torchmetrics.image")
    torchmetrics_fid = types.ModuleType("torchmetrics.image.fid")
    torchmetrics_inception = types.ModuleType("torchmetrics.image.inception")
    torchmetrics_fid.FrechetInceptionDistance = object
    torchmetrics_inception.InceptionScore = object

    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda iterable, **kwargs: iterable

    dependencies = {
        "torch": torch,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "torchvision": torchvision,
        "torchvision.datasets": torchvision_datasets,
        "torchvision.transforms": torchvision_transforms,
        "torchvision.io": torchvision_io,
        "torchmetrics": torchmetrics,
        "torchmetrics.image": torchmetrics_image,
        "torchmetrics.image.fid": torchmetrics_fid,
        "torchmetrics.image.inception": torchmetrics_inception,
        "tqdm": tqdm,
        "pandas": pandas,
    }

    with patch.dict(sys.modules, dependencies):
        sys.modules.pop("evaluate", None)
        return importlib.import_module("evaluate")


class GeneratedImageLoaderTests(unittest.TestCase):
    def test_generated_loader_parallelizes_decode_and_pins_cuda_memory(self):
        evaluator = import_evaluator()
        paths = [Path("first.png"), Path("second.png")]

        loader = evaluator.generated_loader(
            paths,
            batch_size=256,
            num_workers=4,
            device="cuda",
        )

        self.assertEqual(len(loader.dataset), 2)
        self.assertEqual(loader.dataset[0].path, "first.png")
        self.assertEqual(loader.kwargs["batch_size"], 256)
        self.assertEqual(loader.kwargs["num_workers"], 4)
        self.assertTrue(loader.kwargs["pin_memory"])
        self.assertTrue(loader.kwargs["persistent_workers"])
        self.assertEqual(loader.kwargs["prefetch_factor"], 4)

    def test_generated_loader_disables_worker_only_options_without_workers(self):
        evaluator = import_evaluator()

        loader = evaluator.generated_loader(
            [Path("only.png")],
            batch_size=1,
            num_workers=0,
            device="cpu",
        )

        self.assertFalse(loader.kwargs["pin_memory"])
        self.assertNotIn("persistent_workers", loader.kwargs)
        self.assertNotIn("prefetch_factor", loader.kwargs)


class FidConfigurationTests(unittest.TestCase):
    def test_create_fid_preserves_real_statistics_across_resets(self):
        evaluator = import_evaluator()
        created = []

        class RecordingFid:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created.append(self)

            def to(self, device):
                self.device = device
                return self

        with patch.object(evaluator, "FrechetInceptionDistance", RecordingFid):
            fid = evaluator.create_fid("cuda")

        self.assertIs(fid, created[0])
        self.assertEqual(fid.device, "cuda")
        self.assertEqual(fid.kwargs["feature"], 2048)
        self.assertFalse(fid.kwargs["normalize"])
        self.assertFalse(fid.kwargs["reset_real_features"])


class ArgumentTests(unittest.TestCase):
    def test_parse_args_uses_throughput_oriented_defaults(self):
        evaluator = import_evaluator()

        with patch.object(sys, "argv", ["evaluate.py", "--generated_dir", "samples"]):
            args = evaluator.parse_args()

        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.num_workers, 4)
        self.assertFalse(args.overwrite)

    def test_parse_args_accepts_a_per_sampler_image_limit(self):
        evaluator = import_evaluator()

        with patch.object(
            sys,
            "argv",
            ["evaluate.py", "--generated_dir", "samples", "--max_images", "2000"],
        ):
            args = evaluator.parse_args()

        self.assertEqual(args.max_images, 2000)


class GeneratedPathSelectionTests(unittest.TestCase):
    def test_generated_image_paths_limits_each_sampler_to_the_first_sorted_images(self):
        evaluator = import_evaluator()
        sampler_dir = Path("samples/ddim")

        with patch.object(
            Path,
            "glob",
            return_value=[
                sampler_dir / "batch_000" / "003.png",
                sampler_dir / "batch_000" / "001.png",
                sampler_dir / "batch_000" / "002.png",
            ],
        ):
            paths = evaluator.generated_image_paths(sampler_dir, max_images=2)

        self.assertEqual([path.name for path in paths], ["001.png", "002.png"])


class ExperimentMetadataTests(unittest.TestCase):
    def test_parse_experiment_metadata_extracts_cfg_scale_and_sampling_steps(self):
        evaluator = import_evaluator()

        metadata = evaluator.parse_experiment_metadata("ddim_step50_scale75")

        self.assertEqual(metadata, {"cfg_scale": 7.5, "sampling_steps": 50})

    def test_parse_experiment_metadata_keeps_the_interval_scale_numeric(self):
        evaluator = import_evaluator()

        metadata = evaluator.parse_experiment_metadata(
            "ddim_step50_scale7.5_interval0.19-1.61")

        self.assertEqual(metadata, {"cfg_scale": 7.5, "sampling_steps": 50})

    def test_parse_generated_label_extracts_label_from_image_path(self):
        evaluator = import_evaluator()

        label = evaluator.parse_generated_label(Path("batch_000/000123_label_7.png"))

        self.assertEqual(label, 7)


class EvaluationResultTests(unittest.TestCase):
    def test_evaluate_sampler_includes_sampling_configuration_and_conditional_fid(self):
        evaluator = import_evaluator()

        class FakeFid:
            def compute(self):
                return 12.5

            def reset(self):
                pass

        class FakeInception:
            def __init__(self, **kwargs):
                pass

            def to(self, device):
                return self

            def update(self, images):
                pass

            def compute(self):
                return 3.0, 0.1

        class FakeClassFid:
            def __init__(self, score):
                self.score = score

            def compute(self):
                return self.score

            def reset(self):
                pass

        with (
            patch.object(
                evaluator,
                "generated_image_paths",
                return_value=[Path("000000_label_0.png")],
            ),
            patch.object(evaluator, "generated_loader", return_value=[]),
            patch.object(evaluator, "InceptionScore", FakeInception),
        ):
            result = evaluator.evaluate_sampler(
                sampler="ddim_step50_scale75",
                generated_dir=Path("samples"),
                fid=FakeFid(),
                class_fids=[FakeClassFid(score) for score in range(10)],
                batch_size=1,
                num_workers=0,
                device="cpu",
                classifier=object(),
            )

        self.assertEqual(result["cfg_scale"], 7.5)
        self.assertEqual(result["sampling_steps"], 50)
        self.assertEqual(result["cfid"], 4.5)


class ExperimentDiscoveryTests(unittest.TestCase):
    def test_discover_experiment_dirs_returns_every_directory_with_generated_batches(self):
        evaluator = import_evaluator()

        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_dir = Path(temporary_directory)
            for experiment in ("ddim_step50_scale1", "cfgpp_step50_scale06"):
                image_dir = generated_dir / experiment / "batch_000"
                image_dir.mkdir(parents=True)
                (image_dir / "000.png").touch()
            (generated_dir / "notes").mkdir()
            (generated_dir / "empty_experiment").mkdir()

            experiments = evaluator.discover_experiment_dirs(generated_dir)

        self.assertEqual(
            [(name, path.name) for name, path in experiments],
            [
                ("cfgpp_step50_scale06", "cfgpp_step50_scale06"),
                ("ddim_step50_scale1", "ddim_step50_scale1"),
            ],
        )

class ResumeTests(unittest.TestCase):
    def test_evaluated_experiments_reports_the_names_already_in_the_results(self):
        evaluator = import_evaluator()
        existing = evaluator.pd.DataFrame([{"sampler": "ddim_step50_scale75"}])

        self.assertEqual(
            evaluator.evaluated_experiments(existing), {"ddim_step50_scale75"}
        )

    def test_evaluated_experiments_is_empty_when_overwriting(self):
        evaluator = import_evaluator()
        existing = evaluator.pd.DataFrame([{"sampler": "ddim_step50_scale75"}])

        self.assertEqual(evaluator.evaluated_experiments(existing, overwrite=True), set())

    def test_evaluated_experiments_is_empty_without_previous_results(self):
        evaluator = import_evaluator()

        self.assertEqual(evaluator.evaluated_experiments(evaluator.pd.DataFrame()), set())

    def test_discover_experiment_dirs_never_lists_batches_of_skipped_experiments(self):
        evaluator = import_evaluator()

        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_dir = Path(temporary_directory)
            for experiment in ("ddim_step50_scale75", "ddim_step50_scale20"):
                image_dir = generated_dir / experiment / "batch_000"
                image_dir.mkdir(parents=True)
                (image_dir / "000.png").touch()

            listed = []
            with patch.object(
                evaluator,
                "generated_image_paths",
                side_effect=lambda path, **kwargs: listed.append(path.name) or [path],
            ):
                experiments = evaluator.discover_experiment_dirs(
                    generated_dir, skip={"ddim_step50_scale75"}
                )

        self.assertEqual([name for name, _ in experiments], ["ddim_step50_scale20"])
        self.assertEqual(listed, ["ddim_step50_scale20"])

    def test_discover_experiment_dirs_returns_nothing_when_all_are_skipped(self):
        evaluator = import_evaluator()

        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_dir = Path(temporary_directory)
            image_dir = generated_dir / "ddim_step50_scale75" / "batch_000"
            image_dir.mkdir(parents=True)
            (image_dir / "000.png").touch()

            experiments = evaluator.discover_experiment_dirs(
                generated_dir, skip={"ddim_step50_scale75"}
            )

        self.assertEqual(experiments, [])

    def test_load_existing_results_is_empty_before_the_first_run(self):
        evaluator = import_evaluator()

        with tempfile.TemporaryDirectory() as directory:
            existing = evaluator.load_existing_results(
                Path(directory) / "evaluation_results.csv"
            )

        self.assertTrue(existing.empty)

    def test_write_evaluation_results_appends_after_the_existing_rows(self):
        evaluator = import_evaluator()
        existing = evaluator.pd.DataFrame(
            [{"sampler": "ddim_step50_scale75", "fid": 46.9}]
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation_results.csv"
            evaluator.write_evaluation_results(
                [{"sampler": "ddim_step50_scale7.5_interval0.19-1.61", "fid": 30.1}],
                output,
                existing,
            )
            written = evaluator.pd.read_csv(output)

        self.assertEqual(
            list(written["sampler"]),
            ["ddim_step50_scale75", "ddim_step50_scale7.5_interval0.19-1.61"],
        )


class ProgressBarTests(unittest.TestCase):
    def test_progress_batches_labels_batch_work_without_leaving_stale_bars(self):
        evaluator = import_evaluator()
        calls = []

        def recording_tqdm(iterable, **kwargs):
            calls.append((iterable, kwargs))
            return iterable

        with patch.object(evaluator, "tqdm", recording_tqdm):
            batches = ["first", "second"]
            self.assertIs(evaluator.progress_batches(batches, "Evaluating ddim"), batches)

        self.assertEqual(calls, [
            (batches, {
                "desc": "Evaluating ddim",
                "unit": "batch",
                "leave": False,
            }),
        ])

    def test_evaluate_samplers_does_not_create_an_outer_progress_bar(self):
        evaluator = import_evaluator()
        evaluated_samplers = []
        class_fids = object()

        def evaluate_sampler(*, sampler, **kwargs):
            evaluated_samplers.append(sampler)
            self.assertIs(kwargs["class_fids"], class_fids)
            return {"sampler": sampler}

        with (
            patch.object(
                evaluator,
                "tqdm",
                side_effect=AssertionError("Sampler loop must not create a progress bar"),
            ),
            patch.object(evaluator, "evaluate_sampler", side_effect=evaluate_sampler),
        ):
            results = evaluator.evaluate_samplers(
                ["ddim", "cfgpp"],
                generated_dir=Path("samples"),
                fid=object(),
                class_fids=class_fids,
                batch_size=256,
                num_workers=4,
                device="cuda",
                max_images=1000,
                classifier=object(),
            )

        self.assertEqual(evaluated_samplers, ["ddim", "cfgpp"])
        self.assertEqual(results, [{"sampler": "ddim"}, {"sampler": "cfgpp"}])

    def test_evaluate_samplers_reports_accumulated_results_after_each_experiment(self):
        evaluator = import_evaluator()
        saved_results = []

        def evaluate_sampler(*, sampler, **kwargs):
            return {"sampler": sampler}

        with patch.object(evaluator, "evaluate_sampler", side_effect=evaluate_sampler):
            evaluator.evaluate_samplers(
                ["ddim", "cfgpp"],
                generated_dir=Path("samples"),
                fid=object(),
                class_fids=object(),
                batch_size=256,
                num_workers=4,
                device="cuda",
                on_result=lambda results: saved_results.append(list(results)),
                classifier=object(),
            )

        self.assertEqual(
            saved_results,
            [
                [{"sampler": "ddim"}],
                [{"sampler": "ddim"}, {"sampler": "cfgpp"}],
            ],
        )


class LocalCacheTests(unittest.TestCase):
    def test_prepare_evaluation_directories_caches_generated_and_real_data(self):
        evaluator = import_evaluator()
        copied_paths = [Path("/content/tmp/generated"), Path("/content/tmp/real")]

        with patch.object(
            evaluator,
            "copy_directory_to_cache",
            side_effect=copied_paths,
            create=True,
        ) as copy_directory:
            generated_dir, real_dir = evaluator.prepare_evaluation_directories(
                Path("/content/drive/generated"),
                Path("/content/drive/data"),
                "/content/tmp",
            )

        self.assertEqual(generated_dir, copied_paths[0])
        self.assertEqual(real_dir, copied_paths[1])
        self.assertEqual(
            copy_directory.call_args_list,
            [
                ((Path("/content/drive/generated"), Path("/content/tmp"), "generated"), {}),
                ((Path("/content/drive/data"), Path("/content/tmp"), "real"), {}),
            ],
        )


if __name__ == "__main__":
    unittest.main()