import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import efficientnet_v2_m
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch.nn as nn

# ─────────────────────────────────────────────────────
# 1. Ingest Configuration File
# ─────────────────────────────────────────────────────
CONFIG_PATH = "config.py"
print(f"\n📂 Loading parameters from {CONFIG_PATH}...")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

paths = config.get("paths", {})
BASE_DIR = os.path.expanduser(paths.get("base_dir", "."))

TEST_DIR = os.path.join(BASE_DIR, paths.get("test_dir", "test_data"))
# For testing, the "save" path from training becomes our "load" path
MODEL_PATH = os.path.join(BASE_DIR, paths.get("save_model_name", "models/efficientnet/efficientnetv2_m_sliced.pth"))

IMAGE_SIZE = config["hyperparameters"]["image_size"]
CLASSES = config["classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using Device: {device}")

# STRICT NORMALIZATION ONLY: No random rotations or colors for the final exam!
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────────────
# 2. Scanner & Loading Model
# ─────────────────────────────────────────────────────
def get_test_samples(base_dir, classes):
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

print("\n🔍 Scanning Test Data...")
test_samples = get_test_samples(TEST_DIR, CLASSES)

if len(test_samples) == 0:
    print("⚠️ No images found! Make sure the test_data folder is populated.")
    exit()

print(f"🎯 Found {len(test_samples)} test images.")

print("\n⚙️ Loading EfficientNetV2-M Architecture...")
model = efficientnet_v2_m()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))

print(f"📥 Loading trained weights from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

# CRITICAL: Disables dropout layers for clean testing
model.eval() 

# ─────────────────────────────────────────────────────
# 3. Run Inference Once
# ─────────────────────────────────────────────────────
y_true_raw = []
y_pred_raw = []

print("\n🚀 Running AI Inference...")
with torch.no_grad():
    for img_path, label in tqdm(test_samples, desc="Predicting Scripts"):
        try:
            image = Image.open(img_path).convert("RGB")
            tensor = test_transform(image).unsqueeze(0).to(device) 
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            
            y_true_raw.append(label)
            y_pred_raw.append(predicted.item())
        except Exception:
            pass

# ─────────────────────────────────────────────────────
# 4. Generate Standard 12-Class Report
# ─────────────────────────────────────────────────────
report_12c = classification_report(y_true_raw, y_pred_raw, target_names=CLASSES)
acc_12c = accuracy_score(y_true_raw, y_pred_raw) * 100

report_12c_full = (
    "=======================================================\n"
    "🏆 STANDARD 12-CLASS REPORT (EfficientNetV2-M)\n"
    "=======================================================\n"
    f"✅ OVERALL ACCURACY: {acc_12c:.2f}%\n\n"
    f"{report_12c}"
)

with open(os.path.join(BASE_DIR, "m_padded_sliced_model_report_12_class.txt"), "w") as f:
    f.write(report_12c_full)

print("\n" + report_12c_full)

# Save 12-Class Confusion Matrix (Blue)
cm_12c = confusion_matrix(y_true_raw, y_pred_raw)
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cm_12c, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax, shrink=0.8)
ax.set(xticks=np.arange(cm_12c.shape[1]),
       yticks=np.arange(cm_12c.shape[0]),
       xticklabels=CLASSES, yticklabels=CLASSES,
       ylabel='True Label', xlabel='Predicted Label')
ax.set_title(f'12-Class Confusion Matrix (EfficientNetV2-M) — Acc: {acc_12c:.2f}%', fontsize=14, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
# Annotate cells with counts
thresh = cm_12c.max() / 2.0
for i in range(cm_12c.shape[0]):
    for j in range(cm_12c.shape[1]):
        ax.text(j, i, format(cm_12c[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm_12c[i, j] > thresh else 'black', fontsize=9)
plt.tight_layout()
cm_12c_path = os.path.join(BASE_DIR, "m_20ep_sliced_model_confusion_matrix_12_class.png")
plt.savefig(cm_12c_path, dpi=200)
plt.close(fig)
print(f"📊 12-Class Confusion Matrix saved to {cm_12c_path}")

# ─────────────────────────────────────────────────────
# 5. Generate Merged 10-Class Report
# ─────────────────────────────────────────────────────
def map_to_merged(indices):
    merged = []
    for idx in indices:
        if idx in [0, 7]:      # Hindi (0) or Marathi (7)
            merged.append("devanagari_group")
        elif idx in [2, 3]:    # Assamese (2) or Bengali (3)
            merged.append("eastern_nagari_group")
        else:
            merged.append(CLASSES[idx])
    return merged

y_true_merged = map_to_merged(y_true_raw)
y_pred_merged = map_to_merged(y_pred_raw)

unique_labels = sorted(list(set(y_true_merged)))
report_merged = classification_report(y_true_merged, y_pred_merged, target_names=unique_labels)
acc_merged = accuracy_score(y_true_merged, y_pred_merged) * 100

report_merged_full = (
    "=======================================================\n"
    "🏆 MERGED SCRIPT REPORT (Devanagari & Eastern Nagari combined)\n"
    "=======================================================\n"
    f"✅ OVERALL ACCURACY: {acc_merged:.2f}%\n\n"
    f"{report_merged}"
)

with open(os.path.join(BASE_DIR, "m_padded_sliced_model_report_merged_class.txt"), "w") as f:
    f.write(report_merged_full)

print(report_merged_full)

# Save 10-Class (Merged) Confusion Matrix (Blue)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(unique_labels)
y_true_enc = le.transform(y_true_merged)
y_pred_enc = le.transform(y_pred_merged)
cm_10c = confusion_matrix(y_true_enc, y_pred_enc)
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(cm_10c, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax, shrink=0.8)
ax.set(xticks=np.arange(cm_10c.shape[1]),
       yticks=np.arange(cm_10c.shape[0]),
       xticklabels=unique_labels, yticklabels=unique_labels,
       ylabel='True Label', xlabel='Predicted Label')
ax.set_title(f'10-Class Merged Confusion Matrix (EfficientNetV2-M) — Acc: {acc_merged:.2f}%', fontsize=14, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
# Annotate cells with counts
thresh = cm_10c.max() / 2.0
for i in range(cm_10c.shape[0]):
    for j in range(cm_10c.shape[1]):
        ax.text(j, i, format(cm_10c[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm_10c[i, j] > thresh else 'black', fontsize=9)
plt.tight_layout()
cm_10c_path = os.path.join(BASE_DIR, "m_20ep_sliced_model_confusion_matrix_10_class.png")
plt.savefig(cm_10c_path, dpi=200)
plt.close(fig)
print(f"📊 10-Class Confusion Matrix saved to {cm_10c_path}")

print("\n💾 All M-Model reports and confusion matrices have been saved!")
