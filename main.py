import argparse

from torch.utils.data import DataLoader

from src.model import vgg13
from src.data import tw_face
from src import torch_utils


parser = argparse.ArgumentParser(description='Facial Expression Recognition')

parser.add_argument('--checkpoint', default=vgg13.DEFAULT_WEIGHT_PATH)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--batch_size', default=400, type=int)
args = parser.parse_args()

model = vgg13.VGG13.from_state_dict(args.checkpoint, gpu=args.gpu)
preprocess = vgg13.get_preprocess()

loader = lambda x, y: DataLoader(
    torch_utils.FaceDataset(x, y, preprocess),
    batch_size=args.batch_size, shuffle=False, num_workers=0
)

data = tw_face.read_data('young')
torch_utils.eval_acc(model, loader(*data), data[1], 'TW face young', gpu=args.gpu)

data = tw_face.read_data('old')
torch_utils.eval_acc(model, loader(*data), data[1], 'TW face old', gpu=args.gpu)
