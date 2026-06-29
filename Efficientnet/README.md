## Environment Setup
Activate your conda or virtual Environment with python=3.12 (or similar) installed and do the following:
```bash
pip install -r requirements.txt
```

### EfficientNet Model
This folder uses the EfficientNetV2-M architecture for Script Identification. 
The base trained model is automatically downloaded from GitHub releases via the `efficientnet_identifier.py` script.
Following are the 12 languages it is typically trained on: Hindi, English, Gujarati, Punjabi, Marathi, Kannada, Telugu, Tamil, Malayalam, Assamese, Bengali, and Odia.

## Configuration
All hyperparameters and paths are defined in `config.py` (which acts as a JSON configuration file). 
You must ensure `paths` and `hyperparameters` are set appropriately:
```python
{
    "paths": {
        "base_dir": "path/to/base",
        "train_dir": "train_data",
        "test_dir": "test_data",
        "save_model_name": "models/efficientnet/efficientnetv2_m_sliced.pth"
    },
    "hyperparameters": {
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 0.0001,
        "image_size": 480,
        ...
    },
    "classes": [ ... ]
}
```

## Training
To start the training:
```bash
python train.py
```
During training, the model uses dynamic data augmentation (rotations, color jitter, affine transformations). The script automatically saves the checkpoint whenever the validation accuracy improves.

## Testing 
To test your trained model on the `test_data` directory:
```bash
python test.py
```
The code will output class-wise accuracy along with macro precision, recall, and F1-score for the standard 12 classes, as well as a merged 10-class report (where Devanagari and Eastern Nagari scripts are grouped). Additionally, confusion matrices will be saved as images.

## Inference
Script detection can be done using `infer.py` on a single image as input.

```bash
python infer.py --image_path demo_images/D_image_149_9.jpg
# {'predicted_class': 'Assamese'}
```
Simply replace `demo_images/D_image_149_9.jpg` with your image path.

To process a batch of images:
```bash
python infer.py --image_dir demo_images/ --batch
# predictions.csv
```
Using the `--image_dir` and `--batch` arguments, predictions can be made on a batch of images, and the output is saved to `prediction.csv`.

## Evaluation
After inference for a batch of images and obtaining the predictions in a csv file, use this for calculating the accuracy for the language predictions against the actual labels. 

The test CSV should contain two columns: **Filepath** (the image file name/path) and **Language_test** (the ground truth language labels).

```bash
python evaluation.py <path/to/test.csv> <path/to/prediction.csv>
``` 

## API
Start the localhost FastAPI server:
```bash
uvicorn app:app --reload
```

To get the prediction for an image using the EfficientNet model:
- **METHOD:** POST
- **URL:**
```bash
http://127.0.0.1:8000/predict_lang/
```
- **Parameters:** `file`: image_file to be classified

The model will predict the language and return a JSON response.
