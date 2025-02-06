# Script Identification for Indian Language Scene Text
This repository contains scripts and models for identifying scripts in text using different machine learning approaches. The project is structured into three main sections, each implementing a different method for script identification: CLIP, CRNN, and ViT (Vision Transformer).

## Overview

This repository provides implementations for three different script identification methods:

**CLIP**: A contrastive language-image pre-training model for script identification.\
**CRNN**: A Convolutional Recurrent Neural Network-based approach for recognizing scripts in text images.\
**ViT**: A Vision Transformer-based model for script identification tasks.
Each method has its own folder with specific scripts for training, testing, and inference, as well as web app deployments (via FastAPI for CLIP and CRNN). All models are compatible with Python environments, and each method has its own dependencies listed in the respective requirements.txt file

## Installation 
To get started, clone the repository and install the necessary dependencies for the respective method you wish to use:

### Clone the repositoty
```
git clone https://github.com/Bhashini-IITJ/ScriptIdentification
cd ScriptIdentification
``` 

### Install dependencies for the desired model (e.g., CLIP, CRNN, or ViT):

#### [Installtion for CRNN](CRNN/README.md#installation)
#### [Installtion for CLIP](clip/README.md#installation)
#### [Installtion for ViT](Vit/README.md#environmentsetup)

### Usage
Usage of each model can be found in their respective directory.

#### [CRNN Usage](CRNN/README.md#inference)
#### [CLIP Usage](clip/README.md#inference)
#### [ViT Usage](Vit/README.md#inference)

## Acknowledgements
We would like to express our gratitude to the authors and contributors of the following repositories for their valuable contributions to this project:

[CLIP](https://github.com/openai/CLIP): We acknowledge OpenAI for the CLIP model, which provided the foundation for our script identification approach using contrastive language-image pre-training.\
[CRNN](https://github.com/GitYCC/crnn-pytorch): Thanks to the creators of the CRNN model, which served as a core component for the script identification system based on Convolutional Recurrent Neural Networks.\
[ViT](https://github.com/lucidrains/vit-pytorch): Special thanks to the developers behind the ViT model, whose implementation of Vision Transformers greatly influenced our approach to script identification.
