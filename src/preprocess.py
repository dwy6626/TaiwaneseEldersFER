from enum import Enum


import numpy as np
import torch


def crop_face(img, bbox):
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]


class NoFaceMessage(Enum):
    NOFACE = 0
    ONE_FACE = 1
    TWO_FACE_WITH_BBOX = 20
    TWO_FACE_WITH_PREVIOUS = 21
    TWO_FACE_BUT_CANNNOT_DETERMINE = 2
    FACE_BUT_THRESHOLDED = -1


class Detector:
    def __init__(self, threshold=0.9, image_size=160, gpu=False):
        from facenet_pytorch import MTCNN
        self.detector = MTCNN(
            image_size=image_size,
            thresholds=[threshold, threshold, threshold],
            post_process=False,
            select_largest=False,
            device='cuda' if gpu else 'cpu'
        )
        self.threshold = threshold

        self.message = NoFaceMessage

    def detect(self, imgs, save_path=None, bbox=None):

        # Detect faces
        with torch.no_grad():
            results = self.detector.detect(imgs)

        # bbox
        batch_boxes = []
        batch_probs = []
        for b, p in zip(*results):
            if b is not None:
                b = b[0]
                b[2:] = np.ceil(b[2:])
                b[:2] = np.floor(b[:2])
                b[b < 0] = 0
                batch_boxes.append(b.astype(int))
                batch_probs.append(p[0])
            else:
                batch_boxes.append(None)
                batch_probs.append(None)

        return batch_boxes, batch_probs
