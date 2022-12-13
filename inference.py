import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pytesseract
from PIL import Image, ImageDraw, ImageFont

import torch
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification, AdamW

import train 

def inference():

    print("Running inference.py...")

    MODEL_DIR = os.environ["MODEL_DIR"]
    #MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
    #MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
    #MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
    #MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)

    imageName = os.environ["IMAGE_NAME"] 
    print(" imageName", imageName)

    # Load, read and normalize training data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMv3ForSequenceClassification.from_pretrained("saved_model")
    model.to(device);

    query = 'document-classification-dataset/email/doc_000042.png'
    image = Image.open(query).convert("RGB")
    encoded_inputs = processor(image, return_tensors="pt").to(device)
    outputs = model(**encoded_inputs)
    preds = torch.softmax(outputs.logits, dim=1).tolist()[0]
    pred_labels = {label:pred for label, pred in zip(train.label2idx.keys(), preds)}
    print(pred_labels)
    print(image)


    print("Completing train.py...")
 
if __name__ == "__main__":
    inference()
