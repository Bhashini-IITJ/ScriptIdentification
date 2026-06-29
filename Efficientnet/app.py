from fastapi import FastAPI, File, UploadFile
import torch
import tempfile
import os
import json
from efficientnet_identifier import EfficientNetIdentifier

# Define the classes (modify this list based on your actual training classes)
CLASSES = ["hindi", "english", "assamese", "bengali", "gujarati", "kannada", "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu"]

# Load config to get image size
try:
    with open('config.py', 'r') as f:
        config = json.load(f)
    image_size = config.get("hyperparameters", {}).get("image_size", 480)
except Exception:
    image_size = 480

# Initialize the identifier
# This will download the model if it's not present at the checkpoint path
checkpoint_path = "models/efficientnet/efficientnetv2_m_sliced.pth"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

identifier = EfficientNetIdentifier(
    checkpoint_path=checkpoint_path,
    classes=CLASSES,
    image_size=image_size,
    device=device
)

# Initialize the FastAPI app
app = FastAPI()

@app.post("/predict_lang/")
async def classify_image(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    
    # We need to save it to a temporary file because `identify` expects a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_data)
        temp_file_path = temp_file.name

    try:
        # Classify the image using the model
        predicted_label = identifier.identify(temp_file_path)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # Return the result as a JSON response
    return {
        "image_path": file.filename,
        "predicted_label": predicted_label
    }
