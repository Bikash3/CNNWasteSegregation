# Waste Segregation
> This project implements an image classification pipeline to sort common waste materials into categories using a convolutional neural network. The goal is to improve recycling efficiency and reduce landfill waste by automating material sorting.

## Table of Contents
* [General Info](#general-information)
* [Stepwise Process](#stepwise-process)
  * [Step 1: Objective](#step-1---objective)
  * [Step 2: Data Understanding](#step-2---data-understanding)
  * [Step 3: Data Preparation](#step-3---data-preparation)
  * [Step 4: Model Building and Evaluation](#step-4---model-building-and-evaluation)
  * [Step 5: Data Augmentation (optional)](#step-5---data-augmentation-optional)
  * [Step 6: Conclusions and Insights](#step-6---conclusions-and-insights)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information
> This repository contains code and documentation for a convolutional neural network-based waste classification task. The solution processes image folders where each folder corresponds to one waste category and trains a classifier to predict the correct category for new images.

## Stepwise Process

### Step 1 - Objective
- Build a reliable image classification workflow to categorize waste into classes:
  1. Food_Waste
  2. Metal
  3. Paper
  4. Plastic
  5. Other
  6. Cardboard
  7. Glass
- Deliver an end-to-end pipeline: data loading, preprocessing, training, evaluation, and optional augmentation.

### Step 2 - Data Understanding
- Dataset structure: one folder per class, images inside each folder.
- Expected variations: illumination, resolution, backgrounds, partial occlusions.
- Key task: ensure per-image label correctness and inspect class balance.

### Step 3 - Data Preparation
1. Load images from directory using a controlled loader.
2. Convert images to consistent RGB format and normalize pixel values.
3. Resize images to a common target size (example used: 128x128).
4. Create a labeled dataset and verify counts per class.
5. Encode class labels as integer indices.
6. Split data into training and validation sets (example split: 70% train / 30% validation) with stratification to preserve class distribution.
7. Visual checks:
   - Plot class distribution bar chart.
   - Display sample images from each class.

### Step 4 - Model Building and Evaluation
1. Model architecture (example used):
   - Three convolutional blocks (Conv2D → MaxPooling → BatchNormalization).
   - Flatten → Dense → Dropout → Output Dense with softmax for 7 classes.
2. Compile configuration:
   - Loss: sparse categorical crossentropy
   - Optimizer: Adam
   - Metric: accuracy
3. Training:
   - Use callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
   - Example: train for several epochs and restore best weights.
4. Evaluation:
   - Plot training and validation accuracy and loss.
   - Report final validation accuracy and inspect confusion between classes.
   - Save best model artifact (example: best_waste_model.keras).

### Step 5 - Data Augmentation (optional)
- Implement an augmentation pipeline (flip, rotation, zoom, brightness, etc.) to improve generalization and address class imbalance.
- Retrain the model on augmented data and compare metrics against the base model.

### Step 6 - Conclusions and Insights
- Summarize model performance and practical observations:
  - Most/least confused classes.
  - Effect of augmentation on accuracy and robustness.
  - Recommendations for production use (more data, higher resolution, domain-specific augmentation).

## Technologies Used
- Python
- NumPy, pandas
- Matplotlib, Seaborn
- Pillow (PIL)
- TensorFlow / Keras
- scikit-learn

## Acknowledgements
- Project prepared as part of course exercises.
- Reference documentation:
  - https://www.tensorflow.org/
  - https://scikit-learn.org/
  - https://matplotlib.org/

## Contact
### Created by  
  * Bikas Sarkar
