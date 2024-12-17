# ResNet50 CIFAR-100 Classification 

This repository implements a deep learning model for classifying the CIFAR-100 dataset using a fine-tuned ResNet50 model in PyTorch. The project focuses on enhancing model performance through advanced data augmentation, early stopping, and learning rate scheduling.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
  - [Data Augmentation](#data-augmentation)
  - [Training Script](#training-script)
- [Evaluation](#evaluation)
  - [Confusion Matrix](#confusion-matrix)
  - [Classification Report](#classification-report)
- [Installation](#installation)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction
The CIFAR-100 dataset contains 60,000 32x32 color images across 100 classes, with 600 images per class. This project aims to classify these images using a fine-tuned ResNet50 model, leveraging data augmentation and regularization techniques to achieve high accuracy and prevent overfitting.

## Dataset
The CIFAR-100 dataset includes:
- 50,000 training images
- 10,000 validation images

The dataset is preprocessed using the following transformations:
- **Training Data**:
  - Random horizontal flip
  - Random crop with padding
  - Color jitter (brightness, contrast, saturation, hue)
  - Random rotation
  - Random grayscale
  - Resize to 224x224 (to match ResNet50 input size)
  - Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Validation Data**:
  - Resize to 224x224
  - Normalization (same as above)

## Requirements
Dependencies:
- Python 3.8+
- PyTorch
- Torchvision
- scikit-learn
- Matplotlib
- Seaborn

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Model Architecture
The model uses a pretrained ResNet50 model from PyTorch's `torchvision.models`. Modifications include replacing the fully connected layer with:
- `Linear(in_features, 512)`
- `ReLU()`
- `Dropout(0.5)`
- `Linear(512, 100)` (to match CIFAR-100's 100 classes)

The model is trained using CrossEntropyLoss and an SGD optimizer with momentum and weight decay.

## Training
### Data Augmentation
Data augmentation techniques are applied to improve model generalization, including:
- Random horizontal flips
- Random crops with padding
- Color jitter
- Random rotation
- Random grayscale conversion

### Training Script
The training script includes:
- Early stopping with a patience of 7 epochs
- Learning rate scheduler (ReduceLROnPlateau) to reduce learning rate on plateau
- Training metrics: loss and accuracy

Run the training script:
```bash
python train.py
```

## Evaluation
### Confusion Matrix
A confusion matrix is plotted to visualize the model's performance across all 100 classes.

### Classification Report
A classification report is generated to provide precision, recall, and F1-score for each class.

## Installation
Clone the repository and navigate to the project directory:
```bash
git clone <repository-url>
cd <repository-folder>
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Train the model:
   ```bash
   python train.py
   ```
2. Evaluate the trained model:
   ```bash
   python evaluate.py
   ```
3. Visualize results:
   - Confusion matrix and classification report are displayed during evaluation.

## Conclusion
This project demonstrates the use of a fine-tuned ResNet50 model for classifying the CIFAR-100 dataset. By employing advanced data augmentation, learning rate scheduling, and early stopping, the model achieves robust performance while mitigating overfitting.

