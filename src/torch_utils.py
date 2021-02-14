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
    # input PIL and out PIL image
    img = np.array(img)
    h, w, _ = img.shape
    if h > w:
        pad = (h-w) // 2
        pad = [(0, 0), (pad, pad), (0, 0)]
    elif w > h:
        pad = (w-h) // 2
        pad = [(pad, pad), (0, 0), (0, 0)]
    img = np.pad(img, pad)
    return Image.fromarray(img)


def evaluate_batch(func, x, gpu=False):
    with torch.no_grad():
        if gpu:
            x = x.cuda()
        out = func(x)
        if gpu:
            out = out.cpu()
        return out.data.numpy()


def evaluate(func, loader, gpu=False):
    results = []
    for x, _ in loader:
        out = evaluate_batch(func, x, gpu)
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
