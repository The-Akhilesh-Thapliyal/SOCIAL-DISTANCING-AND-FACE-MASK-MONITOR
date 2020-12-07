# SOCIAL DISTANCING AND FACE MASK MONITOR

## Overview

**SOCIAL DISTANCING AND FACE MASK MONITOR** is a computer vision application designed to enhance safety in public spaces by ensuring compliance with social distancing and face mask guidelines. Leveraging deep learning, the application processes live video feeds from a webcam to detect faces, assess mask usage, and monitor social distancing practices. This solution is particularly relevant in scenarios where maintaining public health measures is crucial.

## Features

- **Face Mask Detection**: The application identifies individuals wearing masks and those not wearing masks in real-time, using a Convolutional Neural Network (CNN) trained on a dataset of face images.
- **Social Distancing Monitoring**: Using the YOLO (You Only Look Once) object detection algorithm, the application detects people in the frame and measures the distance between them. If the distance falls below a predefined threshold, the application flags the individuals as "UNSAFE."
- **Real-time Feedback**: The application provides instant visual feedback by overlaying bounding boxes and labels on detected faces. Green boxes indicate compliance, while red boxes signal a violation.
- **Snapshot Functionality**: Users can capture and save images of the current video feed for further analysis or record-keeping.
- **User-Friendly Interface**: The application is built with a graphical user interface (GUI) using Tkinter, making it easy to operate.
- **Visual Indicators**: Provides visual feedback with bounding boxes and labels to indicate whether an individual is "SAFE" or "UNSAFE."

## Project Structure

- **data_preprocessing.ipynb**: Jupyter notebook for preprocessing the dataset, including loading images, converting them to grayscale, resizing, and normalizing.
- **model_training.ipynb**: Jupyter notebook for training the CNN model using the preprocessed data. It includes model architecture, compilation, training, and visualization of results.
- **Project_GUI.py**: The main Python script that loads the trained model and runs the GUI application for real-time monitoring.
- **README.md**: Documentation for the project.
- **Dataset**: Folder containing the images used


## Model Information

- **CNN Model**: Trained to distinguish between images with and without masks using a dataset of face images.
- **YOLOv3**: Used for detecting people in the video feed to monitor social distancing.

## Usage

- **Home Page**: The initial interface where you can start the monitoring process.
- **Monitoring Page**: Displays the real-time video feed, highlighting detected faces with bounding boxes and labels ("WITH MASK" in green and "WITHOUT MASK" in red).
- **Snapshot**: Capture and save the current video frame.
- **Exit**: Safely close the application.


## Application Screenshots

### 1. Home Page
The home page serves as the starting point, where users can initiate the monitoring process.

![home_page](https://github.com/user-attachments/assets/2dd4483b-be69-450b-a4d9-8faab6284ad9)

### 2. Real-time Monitoring
In the monitoring view, the application processes the live video feed to detect faces, determine mask usage, and monitor social distancing. The results are displayed in real-time with visual indicators.

![monitoring](https://github.com/user-attachments/assets/e832b428-a620-4fbc-b739-5f05ca243476)

## Model Training Results

### 1. Epochs of Model Training

Below is a screenshot showing the epochs of model training:

![epochs_training](https://github.com/user-attachments/assets/5a502a77-9d01-489d-a93b-209b4048ed68)

### 2. Plotting Training and Validation Loss

Below is a screenshot showing the plot of training and validation loss over epochs:

![training_validation_loss](https://github.com/user-attachments/assets/22357241-5bfb-4ad7-aeac-458c86474eda)

### 3. Plotting Training and Validation Accuracy

Below is a screenshot showing the plot of training and validation accuracy over epochs:

![training_validation_accuracy](https://github.com/user-attachments/assets/c7341dc0-6482-497c-97a7-bd27ee225e85)

### 4. Evaluating the Model on Test Data

Below is a screenshot showing the evaluation of the model on test data, including performance metrics and results:

![test_data_evaluation](https://github.com/user-attachments/assets/b9f79ca2-3d4c-4e0b-98bb-b3fc4ae1e901)

## 🚀 About Me

Hello, I'm Akhilesh Thapliyal! 👋

## Contact

- **Email:** akhilesh.thedev@gmail.com
- **LinkedIn:** www.linkedin.com/in/akhilesh-thapliyal
- **GitHub:** https://github.com/The-Akhilesh-Thapliyal

