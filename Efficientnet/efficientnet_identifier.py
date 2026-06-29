import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_m
import os
import urllib.request
import json

model_info = {
    "efficientnet_v2m_sliced": {
        "path": "models/efficientnet",
        "url": "https://github.com/Er-Mayank-Aggarwal/IndicPhotoOCR/releases/download/efficientnet-model/efficientnetv2_m_sliced_train_data.pth",
        "filename": "efficientnetv2_m_sliced.pth"
    }
}

def download_model_from_release(model_name, save_path):
    """
    Download a model from GitHub releases if it doesn't already exist locally.
    
    Args:
        model_name (str): Key in model_info dict
        save_path (str): Full path where to save the model
    """
    # Skip download if model already exists locally
    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}, skipping download.")
        return

    if model_name not in model_info:
        raise ValueError(f"Model '{model_name}' not found in model_info")
    
    model_data = model_info[model_name]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        url = model_data["url"]
        print(f"Downloading model from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Model downloaded successfully to {save_path}")
    except Exception as e:
        print(f"Error: Could not download model from release: {e}")
        raise

class EfficientNetIdentifier:
    def __init__(self, checkpoint_path, classes, image_size, device='cuda:0', model_name="efficientnet_v2m_sliced"):
        """
        Args:
            checkpoint_path (str): Path to your saved .pth file.
            classes (list): The list of class names exactly as they are in your config.json.
            image_size (int): The image size used during training (e.g., from config).
            device (str): Compute device.
            model_name (str): Model key in model_info dict for downloading from GitHub.
        """
        self.device = device
        self.classes = classes
        
        # Auto-download model from GitHub release
        download_model_from_release(model_name, checkpoint_path)
        
        # 1. Initialize EfficientNetV2-M (Matching your training code exactly)
        self.model = efficientnet_v2_m()
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(self.classes))
        
        # 2. Extract the model state from your custom checkpoint dictionary
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Your training code saves the weights inside the "model_state_dict" key
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
        self.model.to(self.device)
        self.model.eval()

        # 3. Exact Test Transforms from your testing script
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def identify(self, cropped_path, lang_hint="auto", device=None):
        target_device = device if device else self.device
        
        try:
            image = Image.open(cropped_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(target_device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                _, predicted = torch.max(outputs, 1)
                
            return self.classes[predicted.item()]
        except Exception as e:
            print(f"Error identifying {cropped_path}: {e}")
            return None

    def identify_batch(self, cropped_paths, lang_hint="auto", device=None, batch_size=32):
        target_device = device if device else self.device
        results = []
        
        for i in range(0, len(cropped_paths), batch_size):
            batch_paths = cropped_paths[i:i + batch_size]
            batch_tensors = []
            valid_paths = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    batch_tensors.append(self.transform(image))
                    valid_paths.append(path)
                except Exception:
                    # If an image fails to load, append a fallback label to maintain order
                    results.append(self.classes[0]) 
                    
            if not batch_tensors:
                continue
                
            input_batch = torch.stack(batch_tensors).to(target_device)
            
            with torch.no_grad():
                outputs = self.model(input_batch)
                _, predicted = torch.max(outputs, 1)
                
            # Map predictions to class names
            batch_langs = [self.classes[idx.item()] for idx in predicted]
            results.extend(batch_langs)
        return results
