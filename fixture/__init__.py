import numpy as np
from PIL import Image


SAMPLE_IMAGE = Image.fromarray(
    (np.random.rand(150, 150, 3) * 256).astype('uint8')
)


SAMPLE_DATA = ['./fixture/sample.jpg'], [3]
