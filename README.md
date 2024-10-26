# Skin Cancer Prediction using EfficientNet

This project focuses on classifying skin cancer as either **malignant** or **benign** using a pre-trained **EfficientNet** model. The model is fine-tuned and evaluated on a labeled dataset of skin cancer images, leveraging transfer learning to achieve high accuracy.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Overview
In this project, we build a skin cancer classification model using **EfficientNet**, a state-of-the-art convolutional neural network architecture. The goal is to classify skin cancer as either malignant or benign based on medical image data.

The EfficientNet model is fine-tuned using transfer learning to adapt to the specific dataset, and its performance is evaluated using standard classification metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.

## Dataset
The dataset used for this project can be found on Kaggle:  
[Skin Cancer: Malignant vs. Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign/data)

The dataset consists of images of skin cancer labeled as malignant or benign. It includes thousands of images that have been preprocessed and split into training, validation, and test sets.

## Model Architecture

### **EfficientNet**
- A pre-trained **EfficientNet-B0** model is used for this project.
- The final classification layer has been modified to output two classes (malignant or benign).
- Transfer learning is employed to fine-tune the model on the dataset.

## Training and Evaluation
- **Optimizer**: The model is trained using the **Adam optimizer** with a learning rate schedule to adjust the learning rate during training.
- **Loss Function**: **Cross-entropy loss** is used as the loss function.
- **Metrics**: The model is evaluated using accuracy, precision, recall, and F1-score to measure its performance.
- **Training**: The model is trained for 15 epochs with a learning rate of `1e-4` for the fine-tuned layers and a lower learning rate for the base layers.

## Results
The model achieved strong performance in classifying skin cancer images. The evaluation on the test set shows promising results across multiple metrics.

## Future Work
- Investigate further **hyperparameter tuning** to improve classification accuracy.
- Incorporate more diverse medical image datasets to increase the generalization capability of the model.

## License
This project is licensed under the MIT License.
