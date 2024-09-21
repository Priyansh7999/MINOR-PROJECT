# Face Recognition Model

This project implements a face recognition model using the ResNet Convolutional Neural Network (CNN) algorithm. The model efficiently encodes and recognizes faces through a structured approach.

## Project Structure

The project consists of three main components:

1. **encode.py**: 
   - This script encodes all images present in the training folder.
   - It stores the encodings along with the corresponding names in an `output.pkl` file located in the `output` folder.

2. **validate.py**: 
   - This script detects individuals by comparing a given image to the encodings stored in `output.pkl`.
   - It encodes the input image and compares its encoding with the stored encodings, using a tolerance of 0.4 (approximately 60% matching required for recognition).

3. **face_recognize.py**: 
   - This script presents a user interface using PyQt.
   - It provides a live camera feed and includes three buttons:
     - **Take Sample**: Captures 5 photos to store in the database for encoding.
     - **Take Photo**: Captures a single photo of the person and validates it against the stored encodings.
     - **Train Data**: Performs the encoding work on the images.

## Requirements

- Python 3.x
- PyQt
- OpenCV
- TensorFlow or Keras (for the ResNet model)
- Other required libraries

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Priyansh7999/MINOR-PROJECT
   cd MINOR-PROJECT

## Usage
- Prepare your training images by placing them in the training folder.
- Run the encode.py script to generate the encodings:
- Start the face recognition interface
- Use the buttons in the GUI:
  - Take Sample: Click to capture 5 photos of the same individual for encoding.
  - Take Photo: Click to capture a single photo, which will be validated against the stored encodings.
  - Train Data: Click to perform the encoding work on the images in the training folder.
