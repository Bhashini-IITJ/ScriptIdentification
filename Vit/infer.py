from transformers import AutoImageProcessor,ViTForImageClassification,pipeline
from PIL import Image
from datasets import DatasetDict,Dataset,ClassLabel
import torchvision.transforms as transforms
import numpy as np
import csv
import os
import argparse
import requests
from tqdm import tqdm
import zipfile
import time
import glob
from config import infer_config as config

model_info = {
    "hindi": {
        "path": "models/he",
        "url" : "",
        "subcategories": ["hindi", "english"]
    },
    "assamese": {
        "path": "models/hea",
        "url": "https://github.com/adityarathor007/test/releases/download/V1/hea.zip",
        "subcategories": ["hindi", "english", "assamese"]
    },
    "bengali": {
        "path": "models/heb",
        "url" : "",
        "subcategories": ["hindi", "english", "bengali"]
    },
    "gujarati": {
        "path": "models/heg",
        "url" : "",
        "subcategories": ["hindi", "english", "gujarati"]
    },
    "kannada": {
        "path": "models/hek",
        "url" : "",
        "subcategories": ["hindi", "english", "kannada"]
    },
    "malayalam": {
        "path": "models/hem",
        "url" : "",
        "subcategories": ["hindi", "english", "malayalam"]
    },
    "marathi": {
        "path": "models/hema",
        "url" : "",
        "subcategories": ["hindi", "english", "marathi"]
    },
    "meitei": {
        "path": "models/heme",
        "url" : "",
        "subcategories": ["hindi", "english", "meitei"]
    },
    "odia": {
        "path": "models/heo",
        "url" : "",
        "subcategories": ["hindi", "english", "odia"]
    },
    "punjabi": {
        "path": "models/hep",
        "url" : "",
        "subcategories": ["hindi", "english", "punjabi"]
    },
    "tamil": {
        "path": "models/het",
        "url" : "",
        "subcategories": ["hindi", "english", "tamil"]
    },
    "telugu": {
        "path": "models/hetu",
        "url" : "",
        "subcategories": ["hindi", "english", "telugu"]
    },
    "urdu": {
        "path": "models/heu",
        "url" : "",
        "subcategories": ["hindi", "english", "urdu"]
    }
}

pretrained_vit_model = config['pretrained_vit_model']
processor = AutoImageProcessor.from_pretrained(pretrained_vit_model,use_fast=True)



def unzip_file(zip_path, extract_to):

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted files to {extract_to}")




def ensure_model(model_name):
    model_path = model_info[model_name]["path"]
    url = model_info[model_name]["url"] 

    if not os.path.exists(model_path):
        print(f"Model not found locally. Downloading {model_name} from {url}...")

        response = requests.get(url, stream=True)
        zip_path = os.path.join(model_path, "temp_download.zip")

        os.makedirs(model_path, exist_ok=True)

        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_path)

        os.remove(zip_path)

        print(f"Downloaded and extracted to {model_path}")
    
    else:
        print(f"Model folder already exists: {model_path}")
    
    return model_path





def predict(image_path,model_name):
    model_path = ensure_model(model_name)

    vit = ViTForImageClassification.from_pretrained(model_path)
    model= pipeline('image-classification', model=vit, feature_extractor=processor,device=0)

    if image_path.endswith((".png", ".jpg", ".jpeg")):  

        image = Image.open(image_path)
        output = model(image)
        predicted_label = max(output, key=lambda x: x['score'])['label']
        
        print(f"image_path: {image_path}, predicted_label: {predicted_label}\n")
        


def predict_batch(image_dir,model_name,time_show,output_csv="prediction.csv"):
    model_path = ensure_model(model_name)
    vit = ViTForImageClassification.from_pretrained(model_path)
    model= pipeline('image-classification', model=vit, feature_extractor=processor,device=0)

    start_time = time.time()
    results=[]
    image_count=0
    for filename in os.listdir(image_dir):
        
        if filename.endswith((".png", ".jpg", ".jpeg")):  
            img_path = os.path.join(image_dir, filename)
            image = Image.open(img_path)
            

            output = model(image)
            predicted_label = max(output, key=lambda x: x['score'])['label'].capitalize()
            
            results.append({"Filepath": filename, "Language": predicted_label})
            image_count+=1
    
    elapsed_time = time.time() - start_time

    if time_show:
        print(f"Time taken to process {image_count} images: {elapsed_time:.2f} seconds")
    
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Filepath", "Language"])
        writer.writeheader()
        writer.writerows(results)
    
    return output_csv


if __name__ == "__main__":
    # Argument parser for command line usage
    parser = argparse.ArgumentParser(description="Image classification using CLIP fine-tuned model")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--image_dir", type=str, help="Path to the input image directory")
    parser.add_argument("--model_name", type=str, choices=model_info.keys(), help="Name of the model (e.g., hineng, hinengpun, hinengguj)")
    parser.add_argument("--batch", action="store_true", help="Process images in batch mode if specified")
    parser.add_argument("--time",type=bool, nargs="?", const=True, default=False, help="Prints the time required to process a batch of images")

    args = parser.parse_args()


    # Choose function based on the batch parameter
    if args.batch:
        print("hello")
        if not args.image_dir:
            print("Error: image_dir is required when batch is set to True.")
        else:
            result = predict_batch(args.image_dir, args.model_name, args.time)
            print(result)
    else:
        if not args.image_path:
            print("Error: image_path is required when batch is not set.")
        else:
            result = predict(args.image_path, args.model_name)
            print(result)