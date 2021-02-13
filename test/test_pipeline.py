import unittest

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
        x, y = SAMPLE_DATA
        self.test_x = x[0]

    def test_main_pipeline(self):
        torch_transform = vgg13.get_preprocess()
        dataset = torch_utils.FaceDataset(*SAMPLE_DATA, torch_transform)

        out = self.model(torch.unsqueeze(dataset[0][0], 0))
        self.assertEqual(out.shape, (1, 7))

    def test_face_detection(self):
        img = Image.open(self.test_x)
        bboxs, _ = preprocess.Detector().detect([img])
        face_img = preprocess.crop_face(np.array(img), bboxs[0])

        self.assertGreaterEqual(img.size[0], face_img.shape[0])
        self.assertGreaterEqual(img.size[1], face_img.shape[1])

    @unittest.skipUnless(torch.cuda.is_available(), 'gpu unavailable')
    def test_face_detection_gpu(self):
        img = Image.open(self.test_x)
        bboxs, _ = preprocess.Detector(gpu=True).detect([img])
        face_img = preprocess.crop_face(np.array(img), bboxs[0])

        self.assertGreaterEqual(img.size[0], face_img.shape[0])
        self.assertGreaterEqual(img.size[1], face_img.shape[1])
