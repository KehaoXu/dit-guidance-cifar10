import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_evaluator import import_evaluator


class Metric:
    def __init__(self, **kwargs):
        pass

    def to(self, device):
        return self

    def update(self, images, **kwargs):
        pass

    def compute(self):
        return 1.0

    def reset(self):
        pass


class Inception(Metric):
    def compute(self):
        return 3.0, 0.1


class ClassAccuracyTests(unittest.TestCase):
    def test_accuracy_normalizes_pngs_and_weights_partial_batches(self):
        evaluator = import_evaluator()
        evaluator.torch = torch
        received_images = []

        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, images):
                received_images.append(images.detach().clone())
                scores = torch.zeros(len(images), 10)
                predicted = (images[:, 0, 0, 0] > 0).long()
                scores[torch.arange(len(images)), predicted] = 1
                return scores

        classifier = Classifier()
        images = torch.zeros(3, 3, 32, 32, dtype=torch.uint8)
        images[1:] = 255
        paths = [Path(f'{index:06d}_label_{index}.png') for index in range(3)]
        with (
            patch.object(evaluator, 'generated_image_paths', return_value=paths),
            patch.object(evaluator, 'generated_loader', return_value=[images[:2], images[2:]]),
            patch.object(evaluator, 'InceptionScore', Inception),
            patch.object(torch.hub, 'load', return_value=classifier),
        ):
            result = evaluator.evaluate_sampler(
                sampler='ddim_step50_scale1', generated_dir=Path('samples'),
                fid=Metric(), class_fids=[Metric() for _ in range(10)],
                batch_size=2, num_workers=0, device='cpu',
            )
        self.assertIn('class_accuracy', result)
        self.assertAlmostEqual(result['class_accuracy'], 2 / 3)
        self.assertFalse(classifier.training)
        self.assertFalse(classifier.anchor.requires_grad)
        self.assertEqual(len(received_images), 2)
        torch.testing.assert_close(received_images[0][0, :, 0, 0],
                                   torch.tensor([-2.429066, -2.418255, -2.221393]))
        torch.testing.assert_close(received_images[0][1, :, 0, 0],
                                   torch.tensor([2.514088, 2.596790, 2.753731]))
        self.assertEqual(images.dtype, torch.uint8)
        self.assertEqual(images[1, 0, 0, 0].item(), 255)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results.csv'
            evaluator.write_evaluation_results([result], output)
            saved = evaluator.pd.read_csv(output)
        self.assertAlmostEqual(saved.loc[0, 'class_accuracy'], 2 / 3)

    def test_experiments_share_one_classifier(self):
        evaluator = import_evaluator()
        classifier = object()
        classifiers = []

        def evaluate_sampler(*, sampler, **kwargs):
            classifiers.append(kwargs.get('classifier'))
            return {'sampler': sampler}

        with (
            patch.object(evaluator, 'create_cifar10_classifier', return_value=classifier,
                         create=True) as create_classifier,
            patch.object(evaluator, 'evaluate_sampler', side_effect=evaluate_sampler),
        ):
            evaluator.evaluate_samplers(
                ['ddim', 'cfgpp'], generated_dir=Path('samples'), fid=object(),
                class_fids=object(), batch_size=2, num_workers=0, device='cpu',
            )
        self.assertEqual(classifiers, [classifier, classifier])
        create_classifier.assert_called_once_with('cpu')


if __name__ == '__main__':
    unittest.main()