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

    
    # Load, read and normalize training data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMv3ForSequenceClassification.from_pretrained("saved_model")
    model.to(device);

    query = os.environ["IMAGE_NAME"] 
    print(" imageName: ", query)
    image = Image.open(query).convert("RGB")
    encoded_inputs = processor(image, return_tensors="pt").to(device)
    outputs = model(**encoded_inputs)
    preds = torch.softmax(outputs.logits, dim=1).tolist()[0]
    pred_labels = {label:pred for label, pred in zip(train.label2idx.keys(), preds)}
    print(pred_labels)
    print(image)


    print("Completing inference.py...")
 
if __name__ == "__main__":
    inference()
