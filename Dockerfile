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


#RUN python3 train.py

#COPY setup.py ./
#RUN python setup.py