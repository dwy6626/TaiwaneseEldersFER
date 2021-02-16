import unittest

import torch
import numpy as np

from src import torch_utils
from src import pipeline
from src.model import vgg13
from fixture import SAMPLE_DATA, NO_FACE_DATA

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.model = vgg13.VGG13()
        x, y = SAMPLE_DATA
        self.test_x = x[0]
        x, y = NO_FACE_DATA
        self.test_no_face_x = x[0]

    def test_main_pipeline(self):
        torch_transform = vgg13.get_preprocess()
        dataset = torch_utils.FaceDataset([self.test_x], torch_transform)

        out = self.model(torch.unsqueeze(dataset[0][0], 0))
        self.assertEqual(out.shape, (1, 7))

    def test_face_detection(self):
        img = pipeline.load_image(self.test_x)
        bboxs, _ = pipeline.Detector().detect([img])
        face_img = pipeline.crop_face(img, bboxs[0])

        self.assertGreaterEqual(img.size[0], face_img.size[0])
        self.assertGreaterEqual(img.size[1], face_img.size[1])

    def test_no_face_detection(self):
        img = pipeline.load_image(self.test_no_face_x)
        with self.assertRaises(pipeline.NoFaceError):
            pipeline.Detector().detect([img])

    @unittest.skipUnless(torch.cuda.is_available(), 'gpu unavailable')
    def test_face_detection_gpu(self):
        img = pipeline.load_image(self.test_x)
        bboxs, _ = pipeline.Detector(gpu=True).detect([img])
        face_img = pipeline.crop_face(img, bboxs[0])

        self.assertGreaterEqual(img.size[0], face_img.size[0])
        self.assertGreaterEqual(img.size[1], face_img.size[1])
