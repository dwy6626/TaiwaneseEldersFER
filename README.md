# Facial Expression Recognition for Taiwanese Elders

(under maintenance)

## Setup

### Python Version

please install Python 3.9 first

### Python packages

```
python -m pip install -r requirements.txt
```

### Download weight

## Run model

```
python main.py --input example_face.png
```

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

- in preparation
