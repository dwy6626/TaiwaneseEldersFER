import argparse
import sys


from torch.utils.data import DataLoader
import torch


from src.model import vgg13
from src.data import tw_face
from src import torch_utils, pipeline, misc


parser = argparse.ArgumentParser(description='Facial Expression Recognition')

parser.add_argument('--checkpoint', default=vgg13.DEFAULT_WEIGHT_PATH)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--batch_size', default=400, type=int)
parser.add_argument('--input')
parser.add_argument('--input_folder')
parser.add_argument('--skip_detection', action='store_true')

args = parser.parse_args()



# inputs
if args.input is not None:
    if args.input_folder is not None:
        print('both input and input_folder are provided, use input by default')
    if args.skip_detection:
        img = pipeline.load_image(args.input)
    else:
        img = pipeline.load_face(args.input, gpu=args.gpu)

    preprocess = vgg13.get_preprocess()
    model = vgg13.VGG13.from_state_dict(args.checkpoint, gpu=args.gpu)

    tensor_img = preprocess(img)
    res = torch_utils.evaluate_batch(model, torch.unsqueeze(tensor_img, 0), gpu=args.gpu)
    misc.display_result([args.input], misc.softmax(res))

elif args.input_folder is not None:
    x = misc.list_image(args.input_folder)
    if args.skip_detection:
        pass
        # TODO:
    else:
        imgs = pipeline.load_faces(args.input_folder, gpu=args.gpu)
        # TODO:

else:
    if tw_face.data_exist():
        print('evaluate the model on Taiwanese faces')
    else:
        print('please download Taiwanese faces data first, or use --input / --input_folder to specify inputs')
        sys.exit(0)


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
