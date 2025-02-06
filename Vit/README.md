## Environment Setup
Activate your conda or virtual Environment with python=3.12 installed and do the following:
```bash
pip install -r requirements.txt
```



### Vit Model's folder trained on BSTD 12 classes can be downloaded from:
    "https://drive.google.com/file/d/1u66s0G60j2tKRUYKiH05XQRmp1YCcwjW/view?usp=sharing"

Following are the languages it is trained on: Hindi, English, Gujarati, Punjabi, Marathi, Kannada, Telugu, Tamil, Malayalam, Assamese, Bengali, and Odia.


## Training
In config.py, specify the desired pretrained ViT model by setting the pretrained_vit_model key within the common_config dictionary.
Further in train_config dictionary pass the dataset which contains images along with specifying the maximum number of images to take from each folder and the number of classes.Also specify the directory where the trained model with graphs will be stored.

To start the training 

```bash
python train.py
```
After training, two folders will be created: one containing all checkpoints and another with the best model saved during the entire training process. For testing, the path to the best model folder will be used."

## Testing 
In config.py, specify the desired pretrained ViT model by setting the pretrained_vit_model key within the common_config dictionary.
Similar to above in the config.py file under test_config pass the images folder path and number of classes. Here you also need to pass the best_model's folder which is to be tested

To test
```bash
python test_crnn.py
```

The code will output class-wise accuracy along with macro precision, recall, and F1-score. Additionally, a confusion matrix will be saved in the best_model's folder

## Inference
To get started, clone the repository, switch to dev branch and in the Vit directory  make "models" directory so that model gets downloaded as we use the inference script. The models can be downloaded directly from github [assets](https://github.com/Bhashini-IITJ/ScriptIdentification/releases/tag/Vit_Models) or via specifying the model name as shown below.

Script detection can be done using ```infer.py``` on a single image as input.

```python
python infer.py --image_path demo_images/D_image_149_9.jpg --model_name assamese
# {'predicted_class': 'assamese'}
```
Simply replace ```demo_images/D_image_149_9.jpg``` with your image path and ```assamese``` with the model name for desired language detection which loads a 3 way classifier for the given language,Hindi and English.

Also can use 10-way classifier which is trained on Hindi, English, Assamese,Bengali,Gujarati,Marathi,Odia,Punjabi,Tamil,Telegu

```python
python infer.py --image_path demo_images/D_image_149_9.jpg --model_name 10C
``` 

To process a batch of images
```python
python infer.py --image_dir demo_images/ --model_name odia --batch
# predictions.csv
```
using the ```image_dir``` and ```--batch``` argument predictions can be made on a batch of images.


## Evaluation
After inference for batch of images and obtaining the predictions in a csv file, use this for calculating the accuracy for the language predictions with the actual labels. 

The test.csv should contain two columns **Filepath** that has path of the images and **Language_test** which has the language labels of those image

```python
python evaluation.py <path/to/test.csv> <path/to/predictions.csv>
``` 



## API
Start the localhost FastAPI server:
uvicorn app:app --reload


To get the prediction for image using the ViT model

- METHOD : POST

- URL
```bash
"http://127.0.0.1:8000/predict_lang/"

```
- Parameters: file: image_file to be classified

The model will predict the language.