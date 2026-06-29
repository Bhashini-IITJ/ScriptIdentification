import os
import json
import torch
import random
import numpy as np

from tqdm import tqdm
from PIL import Image

import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────
# 1. Ingest Configuration File
# ─────────────────────────────────────────────────────
CONFIG_PATH = "config.py"
print(f"\n📂 Loading parameters from {CONFIG_PATH}...")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# The provided config.py didn't have 'paths' so we use get() with defaults
paths = config.get("paths", {})
BASE_DIR = os.path.expanduser(paths.get("base_dir", "."))
TRAIN_DIR = os.path.join(BASE_DIR, paths.get("train_dir", "train_data"))
SAVE_PATH = os.path.join(BASE_DIR, paths.get("save_model_name", "models/efficientnet/efficientnetv2_m_sliced.pth"))

BATCH_SIZE = config["hyperparameters"]["batch_size"]
EPOCHS = config["hyperparameters"]["epochs"]
LEARNING_RATE = config["hyperparameters"]["learning_rate"]
IMAGE_SIZE = config["hyperparameters"]["image_size"]
SEED = config["hyperparameters"]["seed"]
VAL_SPLIT = config["hyperparameters"]["val_split"]
CLASSES = config["classes"]

# ─────────────────────────────────────────────────────
# 2. Environment Setup
# ─────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using Device: {device}")
print(f"📦 Batch Size: {BATCH_SIZE} | 🎯 Target Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"⚠️ Learning Rate Active: {LEARNING_RATE}")

# ─────────────────────────────────────────────────────
# 3. Data Loading Engine
# ─────────────────────────────────────────────────────
def get_samples(base_dir, classes):
    samples = []
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    for root, _, files in os.walk(base_dir):
        parent_folder = os.path.basename(root).lower()
        if parent_folder in class_to_idx:
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
                    samples.append((os.path.join(root, file), class_to_idx[parent_folder]))
    return samples

print("\n🔍 Scanning Dataset Directory...")
all_samples = get_samples(TRAIN_DIR, CLASSES)

if not all_samples:
    print(f"❌ No images found in {TRAIN_DIR}. Please check the path and directory structure.")
    print("Also ensure that your config.py has a 'paths' section defining 'train_dir' if it's not 'train_data'")
    exit(1)

labels = [s[1] for s in all_samples]

train_samples, val_samples = train_test_split(
    all_samples, test_size=VAL_SPLIT, random_state=SEED, stratify=labels
)

# ─────────────────────────────────────────────────────
# 4. Dynamic Data Augmentation Pipeline
# ─────────────────────────────────────────────────────
# Directly resizing instead of PadOrResize as requested
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ScriptDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), label

# We explicitly assign the two different transform pipelines here
train_loader = DataLoader(ScriptDataset(train_samples, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(ScriptDataset(val_samples, transform=val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ─────────────────────────────────────────────────────
# 5. EfficientNetV2-M Architecture & FULL UNFREEZE
# ─────────────────────────────────────────────────────
print("\n⚙️ Downloading/Loading EfficientNetV2-M...")
model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)

print("🔓 UNFREEZING ALL LAYERS for full network optimization...")
for param in model.parameters():
    param.requires_grad = True

model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# ─────────────────────────────────────────────────────
# 6. Safe Checkpoint Auto-Resume
# ─────────────────────────────────────────────────────
start_epoch = 0
best_val_acc = 0.0

# Ensure the save directory exists
os.makedirs(os.path.dirname(os.path.abspath(SAVE_PATH)), exist_ok=True)

if os.path.exists(SAVE_PATH):
    print(f"\n🔄 Checkpoint found at {SAVE_PATH}. Resuming training...")
    checkpoint = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint["best_val_acc"]
else:
    print("\n🆕 No checkpoint found. Starting training from scratch.")

# ─────────────────────────────────────────────────────
# 7. The Protected Training Loop
# ─────────────────────────────────────────────────────
for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"M-Model Epoch {epoch+1}/{EPOCHS}")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_description(f"M-Model Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/(pbar.n+1):.4f} | Train Acc: {100 * correct / total:.2f}%")

    scheduler.step()
    
    # Validation Phase
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"🎯 Validation Accuracy: {val_acc:.2f}%")

    # The crucial safeguard: Only save if the model is getting smarter
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "classes": CLASSES
        }
        torch.save(checkpoint_data, SAVE_PATH)
        print(f"🔥 New Best Full M-Model Saved: {best_val_acc:.2f}%")

print("\n🏁 Process Finished Successfully.")
