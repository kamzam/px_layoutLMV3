# PackageX - Assignment 2 - Kamila Zaman

## Description

This assignement aims to use LayoutLMV3 for document classification of the following dataset :
https://www.kaggle.com/code/ritvik1909/layoutlmv2-document-classification/data


![Alt](/layoutlmv3_architecture.png "Title")

## DataSet Specifications


* Document Classes = 3
* Total datapoint = 165
* Class 0: Email (55 items)
* Class 1: Resume (55 items)
* Class 2: Scientific Publications(55 items)
* Source: Kaggle

## Package Requirements 

1. Python 3.8.9
2. Tesseract
3. Transformers
4. Datasets
5. Pillow
6. Docker 


## Installing Docker
Follow instructions here: 
https://docs.docker.com/desktop/


## Instructions


1. Intall & Build from dockerfile


```
docker build -t DL_Assignment_PX_KamilaZaman:latest -f Dockerfile .
```


Dockerfile Contents
```
FROM python:3.8.9


RUN mkdir my-model
ENV MODEL_DIR=/home/kamila/my-model

ENV env_name $IMAGE_NAME

RUN pip install joblib

WORKDIR usr/src/layoutlmv3


RUN apt-get update
RUN apt-get install -y tesseract-ocr

COPY requirements.txt ./

RUN pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

VOLUME /images_dir


COPY train.py ./
COPY inference.py ./


#RUN python3 train.py

#COPY setup.py ./
#RUN python setup.py

```

2. Run

Input Paramaters:
1. "IMAGE_NAME" - For inference on single image, file name passed as value to this environement variable 
2. "/image_dir" - For Batch Inference use Input image directory to run inference, pass path to mounted volume


```
docker run DL_Assignment_PX_KamilaZaman:latest python3 inference.py
```

1. Download Git Project
2. Open terminal (cmd for windows) in project root directory
3. Run Docker Build Commad 

```
docker build -t DL_Assignment_PX_KamilaZaman:latest -f Dockerfile .
```
4. Image should successfully be built with required environment and model training completed
5. For Inference run the docker container using docker run command

```
docker run 
docker run --rm -it -v %cd%/dataset:/layoutlmv3/img_dir --name=kz_px_container DL_Assignment_PX_KamilaZaman:latest python run_app.py
```

For the 

Input Paramaters specification:
1. "IMAGE_NAME" - For inference on single image, file name passed as value to this environement variable 






## Author

[Kamila Zaman](www.linkedin.com/in/kamila-zaman), [GitHub](https://github.com/abaranovskis-redsamurai)


