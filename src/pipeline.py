from PIL import Image
import numpy as np
import torch


def load_image(path):
    return Image.open(path).convert('RGB')


def load_face(path, gpu=False):
    img = load_image(path)
    bboxs, probs = Detector(gpu=gpu).detect([img])
    print('detected face with confidence: {:.2f}'.format(probs[0]))
    return crop_face(img, bboxs[0])


def load_faces(paths, gpu=False):
    # FIXME: may OOM if image number is too large
    imgs = [load_image(path) for path in paths]
    bboxs, _ = Detector(gpu=gpu).detect(imgs)
    return [crop_face(img, bbox) for img, bbox in zip(imgs, bboxs)]


def crop_face(img, bbox):
    arr = np.array(img)
    arr = arr[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return Image.fromarray(arr)


class NoFaceError(Exception):
    pass


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

    def detect(self, imgs):
        # Detect faces
        with torch.no_grad():
            results = self.detector.detect(imgs)

        # bbox
        batch_boxes = []
        batch_probs = []
        for b, p in zip(*results):
            if b is not None:
                # face with largest confidence
                b = b[0]
                b[2:] = np.ceil(b[2:])
                b[:2] = np.floor(b[:2])
                b[b < 0] = 0
                batch_boxes.append(b.astype(int))
                batch_probs.append(p[0])
            else:
                raise NoFaceError

        return batch_boxes, batch_probs
