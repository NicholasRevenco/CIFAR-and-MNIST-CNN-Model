# CIPHAR and MNIST CNN Models
This repository consolidates two deep learning projects, both focusing on image classification through the application of Convolutional Neural Networks (CNNs). These models have been implemented using TensorFlow and Keras, showcasing how to utilize these libraries to construct, train, and evaluate state-of-the-art image classifiers. Below are detailed descriptions of each project.

## MNIST Dataset:
### Overview:
The MNIST project targets handwritten digit recognition, one of the most classic problems in machine learning. The MNIST dataset comprises 60,000 training and 10,000 test grayscale images, each 28x28 pixels, representing digits from 0 to 9.

### Model Architecture:
The model built for the MNIST dataset leverages a sequence of Conv2D and MaxPooling2D layers, followed by Dense (fully connected) layers. The Conv2D layers are employed to learn spatial hierarchies of features, while the MaxPooling2D layers reduce dimensionality, aiding in reducing overfitting and computational cost. The dense layers bring together these learned features to make the final classification.

### Usage:
Refer to mnist_script.py for the full implementation, instructions, and an example of using the trained model to classify individual handwritten digits.

## CIFAR-10 Dataset:
### Overview:
The CIFAR-10 project is an exploration of a more complex problem: classifying 32x32 color images into one of 10 different classes such as airplanes, automobiles, birds, cats, etc. The dataset consists of 60,000 training images and 10,000 test images.

### Model Architecture:
This project demonstrates a deeper and more sophisticated CNN, including Conv2D layers with ReLU activation, MaxPooling2D layers, Dropout layers for regularization, BatchNormalization layers, and Dense (fully connected) layers with Softmax activation. The integration of Dropout and BatchNormalization aids in the model's robustness, preventing overfitting and accelerating training.

### Usage :
Refer to cifar_script.py for the full implementation and instructions. This script also provides insights into how the model handles more complex visual features and how it can be fine-tuned or extended to work with other similar datasets.

## Getting Started:
- Clone the repository: Clone this repo to your local machine using git clone <repo_url>.
- Install the requirements: Run pip install -r requirements.txt to install the required packages.
- Run the scripts: Execute the scripts for MNIST and CIFAR-10 as needed.
- Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- PIL
These projects offer a hands-on experience with image classification using CNNs. Whether you're an enthusiast looking to get started with deep learning or an experienced practitioner, these projects provide an insightful and practical way to understand and implement cutting-edge models for visual recognition tasks.
