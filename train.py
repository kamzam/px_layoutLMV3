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

    # PRE - PROCESSING

    #Assign label to each folder in provided dataset
    # 3 classes 
    # email -> 0
    # resume -> 1
    # scientific publication -> 2




    #cwdir =  os.getcwd()
    #print(cwdir)
    #dataset_path = cwdir + "\document-classification-dataset"


    #Use Mount for Trainig Dataset
    dataset_path = "/img_dir"
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

    #print(train_dataloader)
    #print(valid_dataloader) 
    # Marked Check

    
    #Build the Model - Hugging Face Distribution 
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base",  num_labels=len(label2idx)
    )
    model.to(device);   
    
    # Models training
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3 #small dataset


    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        training_loss = 0.0
        training_correct = 0
        #put the model in training mode
        model.train()
        for batch in tqdm(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss

            training_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            training_correct += (predictions == batch['labels']).float().sum()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Training Loss:", training_loss / batch["input_ids"].shape[0])
        training_accuracy = 100 * training_correct / len(train_data)
        print("Training accuracy:", training_accuracy.item())  
            
        validation_loss = 0.0
        validation_correct = 0

        # Testing the model on Validation Dataset (15) images
        #Accuracy  = True Negative + True Positves / (TN + TP+ FN+ FP)
        # Here,  validation_correct == True Negative + True Positves 
        for batch in tqdm(valid_dataloader):
            outputs = model(**batch)
            loss = outputs.loss

            validation_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            validation_correct += (predictions == batch['labels']).float().sum()

        print("Validation Loss:", validation_loss / batch["input_ids"].shape[0])
        validation_accuracy = 100 * validation_correct / len(valid_data)
        print("Validation accuracy:", validation_accuracy.item())  

    # Save model
    model.save_pretrained('saved_model/')

    # Record model
   
    print("Completed train.py...")


if __name__ == "__main__":
    train()
