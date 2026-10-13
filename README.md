# CNN-Based Road Damage Detection for Smart City Infrastructure Monitoring

## Overview
This project uses a custom Convolutional Neural Network (CNN) to classify road
images into two classes: normal road and potholes. It is a small educational
deep learning case study implemented with PyTorch.

## Dataset
The dataset contains 681 images: 352 normal-road images and 329 pothole images.
The images were split into training (476), validation (102), and test (103)
sets using stratified splitting with random seed 42.

The dataset is not included in this repository. Place the split folders here:

    data/road_damage_data/
        train/normal/
        train/potholes/
        val/normal/
        val/potholes/
        test/normal/
        test/potholes/

## Model
- Three convolutional layers (32, 64, and 128 channels)
- ReLU activations and max pooling
- Adaptive average pooling
- Fully connected layer and dropout
- One logit with BCEWithLogitsLoss

## Reported test results
These results were measured in the accompanying Colab run on 103 held-out
test images:
- Accuracy: 91.26%
- Precision: 90.20%
- Recall: 92.00%
- F1-score: 91.09%

Results depend on the dataset split and may differ on new road images.
This classifier labels the whole image; it does not draw bounding boxes or
locate damage within an image.

## Setup
Install dependencies:

    pip install -r requirements.txt

Put the dataset into the folder structure above, then run from the project root:

    python -m src.train
    python -m src.evaluate
    python -m src.predict path/to/road_image.jpg

The trained checkpoint is provided at `outputs/best_model.pth`.

## Reference
This project was developed as an educational implementation inspired by the
road-damage CNN repository:
https://github.com/nisabicebekar/road-damage-detection-cnn

The dataset is stored separately to keep this code package small.
