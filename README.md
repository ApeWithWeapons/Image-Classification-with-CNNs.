# Image Classification with Convolutional Neural Networks

Deep learning project classifying 60,000 images across 10 categories 
using a custom CNN architecture trained on the CIFAR-10 dataset.

## Project Overview
I designed and trained a convolutional neural network from scratch to 
classify images into 10 categories. The architecture uses three 
convolutional blocks with batch normalisation and dropout for 
regularisation, data augmentation to improve generalisation, and 
early stopping with learning rate scheduling for efficient training.

## Architecture

| Layer Block | Filters | Key Techniques |
|-------------|---------|----------------|
| Block 1 | 32 | Conv2D, Batch Norm, MaxPool, Dropout 25% |
| Block 2 | 64 | Conv2D, Batch Norm, MaxPool, Dropout 25% |
| Block 3 | 128 | Conv2D, Batch Norm, MaxPool, Dropout 25% |
| Head | 512 | Dense, Batch Norm, Dropout 50%, Softmax |

## Key Findings

| # | Theme | Finding |
|---|-------|---------|
| 1 | Dataset | Perfectly balanced - 5,000 training images per class |
| 2 | Architecture | Increasing filter depth progressively learns simple to complex features |
| 3 | Regularisation | Dropout and data augmentation effectively controlled overfitting |
| 4 | Hardest Classes | Cat and dog most frequently confused due to visual similarity |
| 5 | Early Stopping | Training halted before maximum epochs preventing wasted computation |
| 6 | Transfer Learning | Pretrained models such as VGG16 would achieve 90%+ vs 75-80% from scratch |

## Classes
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

## Training Configuration
- Optimiser: Adam with learning rate 0.001
- Loss: Categorical Crossentropy
- Callbacks: Early Stopping (patience=10), ReduceLROnPlateau (patience=5)
- Data Augmentation: Horizontal flip, width/height shift, rotation
- Batch Size: 64
- Maximum Epochs: 50

## Tools Used
Python, numpy, matplotlib, seaborn, TensorFlow, Keras, scikit-learn

## Dataset
CIFAR-10 is built into Keras and loads automatically with:
keras.datasets.cifar10.load_data()
