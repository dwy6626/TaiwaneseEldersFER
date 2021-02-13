import unittest
import os

import torch

from fixture import SAMPLE_IMAGE
from src.model import vgg13


class TestModel(unittest.TestCase):
    def test_get_model(self):
        model = vgg13.get_model()
        self.assertIsInstance(model, vgg13.VGG13)

    def test_get_model_with_input_size(self):
        model = vgg13.get_model(64)
        self.assertIsInstance(model, vgg13.VGG13)

    @unittest.skipUnless(os.path.exists(vgg13.DEFAULT_WEIGHT_PATH), f'{vgg13.DEFAULT_WEIGHT_PATH} not exist')
    def test_from_state_dict(self):
        model = vgg13.VGG13.from_state_dict(vgg13.DEFAULT_WEIGHT_PATH, gpu=False)
        self.assertIsInstance(model, vgg13.VGG13)
        self.assertEqual(next(model.parameters()).device.type, 'cpu')

    @unittest.skipUnless(os.path.exists(vgg13.DEFAULT_WEIGHT_PATH), f'{vgg13.DEFAULT_WEIGHT_PATH} not exist')
    @unittest.skipUnless(torch.cuda.is_available(), 'gpu unavailable')
    def test_from_state_dict_gpu(self):
        model = vgg13.VGG13.from_state_dict(vgg13.DEFAULT_WEIGHT_PATH, gpu=True)
        self.assertIsInstance(model, vgg13.VGG13)
        self.assertIn('cuda', next(model.parameters()).device.type)

    def test_get_preprocess(self):
        test_size = 80
        preprocess = vgg13.get_preprocess(test_size)
        result = preprocess(SAMPLE_IMAGE)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, test_size, test_size))
