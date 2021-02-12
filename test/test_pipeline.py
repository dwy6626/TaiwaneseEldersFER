import unittest

from torch.utils.data import DataLoader
import torch

from src import torch_utils
from src.model import vgg13
from test.sample import SAMPLE_DATA

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.model = vgg13.VGG13()

    def testPipeline(self):
        x, y = SAMPLE_DATA
        preprocess = vgg13.get_preprocess()
        dataset = torch_utils.FaceDataset(x, y, preprocess)
        out = self.model(torch.unsqueeze(dataset[0][0], 0))
        self.assertEqual(out.shape, (1, 7))
