#!/usr/bin/python3

import platform

print(platform.platform())
import sys

print("Python", sys.version)
import numpy

print("NumPy", numpy.__version__)

import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import pytesseract
from PIL import Image, ImageDraw, ImageFont

import torch
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from transformers import  LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification, AdamW

def train():

    print("Running train.py...")

    # Load directory paths for persisting model

    MODEL_DIR = os.environ["MODEL_DIR"]
    #MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
    #MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)


    #assign label to each folder in provided dataset
    # 3 classes email -> 0, resume -> 1, scientific publication -> 2
    cwdir =  os.getcwd()
    print(cwdir)
    dataset_path = cwdir + "\document-classification-dataset"
    labels = [label for label in os.listdir(dataset_path)]
    idx2label = {v: k for v, k in enumerate(labels)}
    label2idx = {k: v for v, k in enumerate(labels)}
    label2idx       


    
    images = []
    labels = []

    for label in os.listdir(dataset_path):
        images.extend([
            f"{dataset_path}/{label}/{img_name}" for img_name in os.listdir(f"{dataset_path}/{label}")
        ])
        labels.extend([
            label for _ in range(len(os.listdir(f"{dataset_path}/{label}")))
        ])
    data = pd.DataFrame({'image_path': images, 'label': labels})

    train_data, valid_data = train_test_split(data, test_size=0.09, random_state=0, stratify=data.label)
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    print(f"{len(train_data)} training examples, {len(valid_data)} validation examples")
    data.head()
   

    print("Shape of the training data")
    print(train_data.shape)
    print(valid_data.shape)

    # select GPU if available -performance efiiciency

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # LayoutMV3 version pre-processing 
    def encode_training_example(examples):
        images = [Image.open(path).convert("RGB") for path in examples['image_path']]
        encoded_inputs = processor(images, padding="max_length", truncation=True)
        encoded_inputs["labels"] = [label2idx[label] for label in examples["label"]]
        
        return encoded_inputs

        training_features = Features({
            'image': Array3D(dtype="int64", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'token_type_ids': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': ClassLabel(num_classes=len(label2idx), names=list(label2idx.keys())),
        })



    def training_dataloader_from_df(data):
        dataset = Dataset.from_pandas(data)
        
        encoded_dataset = dataset.map(
            encode_training_example, remove_columns=dataset.column_names, features=training_features, 
            batched=True, batch_size=2
        )
        encoded_dataset.set_format(type='torch', device=device)
        dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=4, shuffle=True)
        batch = next(iter(dataloader))
        return dataloader

    # Feature Extraction - dataloader prep 
    feature_extractor = LayoutLMv3FeatureExtractor()
    tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base") #microsoft/layoutlmv3-base-uncased
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)

    train_dataloader = training_dataloader_from_df(train_data)
    valid_dataloader = training_dataloader_from_df(valid_data)
    
    # Models training

    # Save model

    # Record model
    print(train_dataloader)
    print(valid_dataloader)
    print("Completed train.py...")


if __name__ == "__main__":
    train()
