Cat vs Dog Image Classification using Convolutional Neural Networks (CNN)
Project Overview
This project implements a Convolutional Neural Network (CNN) for binary image classification to distinguish between cats and dogs. The model was developed using TensorFlow/Keras and trained on a labeled image dataset. The project includes data preprocessing, dataset cleaning, model training, evaluation, and prediction on new images.
The primary objective of this project is to understand the complete deep learning workflow for image classification, from preparing datasets to deploying a trained model for inference.


Features
Image classification using a custom CNN
Dataset cleaning to identify and remove corrupted images
Image preprocessing and normalization
Model training with TensorFlow/Keras
Early Stopping to prevent overfitting
Model Checkpoint for saving the best-performing model
Prediction on custom images
Visualization of training accuracy and loss
Well-organized project structure suitable for GitHub

Technologies Used
Python 3.12
TensorFlow / Keras
NumPy
Matplotlib
Pillow
Visual Studio Code

Project Structure
Cat-Dog-Image-Classification-CNN/
│
├── src/
│   ├── clean_dataset.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── best_model.keras
│
├── results/
│   ├── accuracy.png
│   └── loss.png
│
├── sample_images/
│   ├── cat.jpg
│   └── dog.jpg
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore


Dataset

The project uses the Dogs vs Cats image dataset.

The dataset should be organized as follows:

dogs-vs-cats-classification/
│
├── train/
│   ├── cats/
│   └── dogs/
│
├── validation/
│   ├── cats/
│   └── dogs/
│
└── test/
    ├── cats/
    └── dogs/



CNN Architecture

The implemented CNN consists of:

Conv2D (32 filters)
MaxPooling2D
Conv2D (64 filters)
MaxPooling2D
Conv2D (128 filters)
MaxPooling2D
Flatten Layer
Dense Layer (128 neurons)
Dropout (0.5)
Output Layer (Sigmoid)


Model Configuration
Parameter	Value
Image Size	128 × 128
Batch Size	32
Epochs	10
Optimizer	Adam
Loss Function	Binary Crossentropy
Output Activation	Sigmoid


Training

Run the following command:

python src/train.py

During training, the script:

Loads the dataset
Normalizes images
Trains the CNN model
Saves the best-performing model
Generates accuracy and loss curves



Prediction

To classify a new image:

python src/predict.py

Enter the full image path when prompted.



Results

The trained model is capable of classifying cats and dogs with high accuracy on the test dataset.

Training graphs are automatically saved in the results/ folder.