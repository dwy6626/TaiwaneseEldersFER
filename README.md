# Facial Expression Recognition for Taiwanese Elders

## Setup

### Python Version

please install Python 3.9 first

### Python packages

```
python -m pip install -r requirements.txt
```

## Run model

```
python main.py --input fixture/sample.jpg
```

```
python main.py --input_folder example_folder
```

p.s. only the face with largest confidence will be evaluated

### Pretrained model

Loading the Pytorch state_dict:

```
--checkpoint weight/best.ckpt
```

There are 2 different pretrained weights

- `best.ckpt` is the cross-dataset model (default)
- `fine_tune.ckpt` is the fine-tuned weights on TW old faces.

## Run Test

```
python -m unittest
```

## Run on Taiwanese Faces

Please request the data from the author of the following paper:

[Tu, Y. Z., Lin, D. W., Suzuki, A., Goh, J. O. S. (2018). East Asian young and older adult perceptions of emotional faces from an age- and sex-fair East Asian facial expression database. *Frontiers in Psychology*, 9. doi: 10.3389/fpsyg.2018.02358](https://www.frontiersin.org/articles/10.3389/fpsyg.2018.02358/)

and put the face images from `pick_faces_628` to the folder `./data`

Also, put the `FaceTake_SubjList.csv` to the project folder

Then, run:

```
python main.py
```

## Reference

- De-Wei Ye, Automatic Facial Expression Recognition for Taiwanese Elders with Deep Convolutional Neural Network,” Master’s thesis, Graduate Institute of Communication Engineering, National Taiwan University, 2020.
