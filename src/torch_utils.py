import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, x, y, preprocess_func, fromarray=False):
        self.data_list = x
        self.y = y
        self.pipeline = preprocess_func
        self.fromarray = fromarray

    def __getitem__(self, i):
        # get file name and load
        p = self.data_list[i]
        if self.fromarray:
            x = Image.fromarray(p).convert('RGB')
        else:
            x = Image.open(p).convert('RGB')

        # pad to square
        x = pad_square(x)
        x = self.pipeline(x)
        return x, self.y[i]

    def __len__(self):
        return self.y.shape[0]


def pad_square(img):
    to_pil = False
    if isinstance(img, Image.Image):
        to_pil = True
        img = np.array(img)

    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    h, w, c = img.shape
    if h > w:
        pad = (h-w) // 2
        pad = [(0, 0), (pad, pad), (0, 0)]
        img = np.pad(img, pad)
    elif w > h:
        pad = (w-h) // 2
        pad = [(pad, pad), (0, 0), (0, 0)]
        img = np.pad(img, pad)
    if to_pil:
        img = Image.fromarray(img)
    return img


def evaluate(func, loader, gpu=1):
    results = []
    with torch.no_grad():
        for x, y in loader:
            if gpu:
                x = x.cuda()
            out = func(x)
            if gpu:
                out = out.cpu()
            out = out.data.numpy()
            results.append(out)

    results = np.concatenate(results)
    return results.reshape(-1, results.shape[1])


def eval_acc(model, loader, truths, name=None, gpu=1):
    model.eval()
    if gpu:
        model.cuda()
    results = evaluate(model, loader, gpu=gpu)
    predictions = np.argmax(results, axis=1)
    acc = np.where(predictions == truths)[0].size / truths.size

    print('{}'.format(name))
    print('Accuracy: {:.2%}'.format(acc))
