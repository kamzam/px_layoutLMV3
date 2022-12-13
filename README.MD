# PackageX - Assignement 2 - Kamila Zaman

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
2. Numpy
3. Pandas
4. Tesseract
5. Transformers
6. Datasets
7. Pillow
8. Docker 


## Installing Docker

Pre-Requisite:
1. Enable Virtualization Technology in your machines BIOS settings
2. Istall WSL for backend linux for docker operation

### Docker Desktop

Follow instructions here: 
https://docs.docker.com/desktop/


## Instructions for project build

If building from project's provided built image kindly follow the steps below:

```
docker load --input dl_assignment_px_kamila_zaman:latest 
```

Once the built image is loaded, use docker run to perform the task as follows:

```
docker run --rm -it -v %cd%/dataset:/layoutlmv3/img_dir  --env-file my-env.txt --name=kz_px_container dl_assignment_px_kamila_zaman:latest python run_app.py
```

Here, 
-v is mounting the dataset directory to the "/img/dir" in container 
--env-file is specifying the image to use for inference. To change the image kindly change the name in my-enc=v.txt

**
-env can be used to pass vale to the IMG_NAME variable for convenience and ese of the docker run statement my-env.txt is used. It is useful design wise as when we have several environemt variables it is way easier to pass values to them from a file than to type it in the command itself.




1. Intall & Build from dockerfile


Build and run docker image from dockerfile in source git project as follows:

Run the following commands.

    1. Clone Git Project
    2. Open terminal (cmd for windows) in project root directory
    3. Run Docker Build Command 
    4. Image should be successfully built with required environment 
   

```
docker build -t dl_assignment_px_kamila_zaman:latest -f Dockerfile .
```


Dockerfile Contents
```
FROM python:3.8.9


RUN mkdir my-model
ENV MODEL_DIR=/home/kamila/my-model

ENV env_name $IMAGE_NAME

RUN pip install joblib

WORKDIR layoutlmv3

VOLUME datavolume

RUN apt-get update
RUN apt-get install -y tesseract-ocr

COPY requirements.txt ./
COPY packages.txt ./

RUN pip install -r requirements.txt

RUN pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html


COPY train.py ./
COPY inference.py ./
COPY run_app.py ./


```

 
    5. For task completion run the docker container using docker run command
  

```
docker run --rm -it -v %cd%/dataset:/layoutlmv3/img_dir  --env-file my-env.txt --name=kz_px_container DL_Assignment_PX_KamilaZaman:latest python run_app.py
```

Here, 
*-v is mounting the dataset directory to the "/img/dir" in container 
*--env-file is specifying the image to use for inference. To change the image kindly change the name in my-enc=v.txt





## Author

[Kamila Zaman](www.linkedin.com/in/kamila-zaman), [GitHub](https://github.com/abaranovskis-redsamurai)


