import unittest

from torch.utils.data import DataLoader
import torch
from PIL import Image
import numpy as np

from src import torch_utils
from src import preprocess
from src.model import vgg13
from test.sample import SAMPLE_DATA

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.model = vgg13.VGG13()

    def testMainPipeline(self):
        x, y = SAMPLE_DATA
        torch_transform = vgg13.get_preprocess()
        dataset = torch_utils.FaceDataset(x, y, torch_transform)

        out = self.model(torch.unsqueeze(dataset[0][0], 0))
        self.assertEqual(out.shape, (1, 7))

    def testFaceDetection(self):
        x, y = SAMPLE_DATA
        img = Image.open(x[0])
        bboxs, _ = preprocess.Detector().detect([img])
        face_img = preprocess.crop_face(np.array(img), bboxs[0])

        self.assertGreaterEqual(img.size[0], face_img.shape[0])
        self.assertGreaterEqual(img.size[1], face_img.shape[1])
