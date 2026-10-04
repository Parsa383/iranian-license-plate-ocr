# Iranian License Plate OCR

An end-to-end deep learning project for detecting and recognizing characters from Iranian license plate images.

The system uses two trained models:

1. **YOLO11n** for detecting individual characters inside a license plate.
2. **CNN classifier** for recognizing each detected character.

The final pipeline receives an image of an Iranian license plate and returns the predicted plate text.

---

## Project Overview

The project is divided into two main machine learning tasks:

### 1. Character Detection

A YOLO object detection model is trained to locate individual characters inside the license plate.

The detector has only one object class:

```text
character
```

Its job is only to answer:

```text
Where are the characters?
```

It does not determine which character was detected.

---

### 2. Character Classification

After YOLO detects the characters, each detected region is cropped and passed to a CNN classifier.

The CNN answers:

```text
Which character is this?
```

The classifier was trained on 31 character classes.

---

## System Pipeline

The complete recognition pipeline is:

```text
License Plate Image
        ↓
YOLO Character Detector
        ↓
Character Bounding Boxes
        ↓
Sort Bounding Boxes Left-to-Right
        ↓
Crop Characters
        ↓
Resize to 32 × 64
        ↓
Convert to Grayscale
        ↓
Normalize Pixel Values
        ↓
CNN Character Classifier
        ↓
Predicted Characters
        ↓
Final License Plate Text
```

---

## Project Structure

```text
iranian-license-plate-ocr/
│
├── data/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── yolo_dataset/
│
├── models/
│   ├── license_plate_character_cnn.keras
│   ├── license_plate_character_detector.pt
│   └── class_mapping.json
│
├── src/
│   ├── read_annotation.ipynb
│   ├── train_character_detector.ipynb
│   └── Inference.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Notebook Description

### `read_annotation.ipynb`

This notebook contains the character classification pipeline.

Main tasks:

- Parse XML annotation files
- Extract character bounding boxes
- Crop individual characters
- Resize characters to `32 × 64`
- Convert images to grayscale
- Normalize pixel values
- Encode character labels
- Prepare train, validation and test datasets
- Build the CNN architecture
- Train the CNN classifier
- Evaluate the model
- Analyze incorrect predictions
- Generate a confusion matrix
- Save the trained CNN model
- Save the class mapping

The trained model is saved as:

```text
models/license_plate_character_cnn.keras
```

The label mapping is saved as:

```text
models/class_mapping.json
```

---

### `train_character_detector.ipynb`

This notebook contains the YOLO character detection pipeline.

Main tasks:

- Read character bounding boxes from XML annotations
- Convert XML bounding boxes to YOLO format
- Generate YOLO labels
- Build the YOLO dataset structure
- Configure the YOLO dataset
- Load a pretrained YOLO11n model
- Fine-tune YOLO on Iranian license plate characters
- Evaluate the detector
- Save the best trained detector

The final detector is stored as:

```text
models/license_plate_character_detector.pt
```

The YOLO model detects only one class:

```text
0 = character
```

---

### `Inference.ipynb`

This notebook contains the final end-to-end inference pipeline.

It loads:

```text
YOLO Character Detector
+
CNN Character Classifier
+
Character Class Mapping
```

The notebook receives the path of a license plate image and performs the following operations automatically:

```text
Image
↓
Character Detection
↓
Bounding Box Sorting
↓
Character Cropping
↓
Preprocessing
↓
CNN Classification
↓
Plate Text
```

---

## CNN Architecture

The character classifier uses a Convolutional Neural Network.

Architecture:

```text
Input: 64 × 32 × 1

Conv2D
↓
MaxPooling2D
↓
Conv2D
↓
MaxPooling2D
↓
Flatten
↓
Dense
↓
Softmax Output
```

The output layer contains:

```text
31 classes
```

The model uses:

```text
Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Metric: Accuracy
```

---

## CNN Performance

The CNN was evaluated on the test dataset.

Final test results:

```text
Test Accuracy: 97.38%
Test Loss: 0.117
```

The model was also evaluated using a confusion matrix to analyze the most common character classification errors.

---

## YOLO Character Detector

The object detector is based on:

```text
YOLO11n
```

A pretrained YOLO11n model was fine-tuned using the license plate character bounding boxes from the dataset.

Training configuration:

```text
Epochs: 3
Image Size: 640
Batch Size: 16
Device: Apple MPS
```

Training was performed using the GPU of an Apple M3 system through PyTorch MPS.

---

## YOLO Validation Results

Final validation results:

```text
Precision: 0.973
Recall: 0.987
mAP50: 0.989
mAP50-95: 0.684
```

These results show that the model can detect most characters in license plate images with high precision and recall.

---

## Bounding Box Format

The original dataset contains bounding boxes in XML format:

```text
xmin
ymin
xmax
ymax
```

For YOLO training, the annotations are converted to:

```text
class_id
x_center
y_center
width
height
```

The coordinates are normalized relative to the image width and height.

Example:

```text
0 0.35 0.52 0.08 0.30
```

Where:

```text
0       = character class
0.35    = normalized x center
0.52    = normalized y center
0.08    = normalized width
0.30    = normalized height
```

---

## Character Preprocessing

Before a detected character is passed to the CNN, the following preprocessing is applied:

```text
Character Crop
↓
Resize to 32 × 64
↓
Convert BGR to Grayscale
↓
Convert to float32
↓
Normalize to range 0–1
↓
Add Channel Dimension
```

The final CNN input shape for one character is:

```text
(64, 32, 1)
```

For multiple detected characters:

```text
(number_of_characters, 64, 32, 1)
```

---

## Installation

Create a Python virtual environment.

Example on macOS/Linux:

```bash
python3 -m venv ai1_env
```

Activate it:

```bash
source ai1_env/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

## Main Libraries

The main libraries used in this project are:

```text
Python
TensorFlow / Keras
PyTorch
Ultralytics YOLO
OpenCV
NumPy
Matplotlib
Scikit-learn
ElementTree
```

---

## Running the Project

Open:

```text
src/Inference.ipynb
```

Make sure the trained models exist:

```text
models/license_plate_character_detector.pt
models/license_plate_character_cnn.keras
models/class_mapping.json
```

Then provide the path of the input license plate image.

Example:

```python
image_path = "/path/to/license_plate.jpg"
```

Run the prediction function:

```python
plate = predict_plate(image_path)

print("Predicted Plate:", plate)
```

The pipeline automatically:

1. Loads the image
2. Detects characters using YOLO
3. Sorts detected characters from left to right
4. Crops each character
5. Preprocesses the character images
6. Classifies them using the CNN
7. Builds the final plate text

---

## Example Prediction

Example predicted character list:

```python
['1', '3', 'ب', '1', '1', '1', '1', '0']
```

The detected characters are ordered based on their horizontal bounding box position.

Because Persian text is right-to-left while numbers are left-to-right, displaying mixed Persian and numeric plate text may require Unicode bidirectional handling.

---

## Dataset

The dataset contains Iranian license plate images and XML annotation files.

Each image has a corresponding XML file containing multiple character annotations.

Example:

```text
00001.jpg
00001.xml
```

Each XML annotation contains:

```text
Character label
Bounding box coordinates
```

The dataset is separated into:

```text
Train
Validation
Test
```

---

## Annotation Issues

A small number of dataset annotations contain invalid coordinates or extremely small images.

During YOLO training, Ultralytics automatically ignored corrupted images or labels.

Examples of detected problems included:

```text
Bounding boxes outside image boundaries
Non-normalized YOLO coordinates
Images smaller than the minimum supported size
```

The majority of the dataset remained valid and was successfully used for training.

---

## Current Limitation

The current system expects the input image to contain the license plate itself.

For example:

```text
License plate image
→ Supported
```

But a full vehicle image such as:

```text
Car image
→ License plate somewhere inside the image
```

is not currently handled automatically.

The existing YOLO detector was trained to detect **characters inside a plate**, not the plate itself.

---

## Future Improvement: License Plate Detection

To support full vehicle images, another object detection model can be added before the current pipeline.

The future architecture would become:

```text
Full Vehicle Image
        ↓
License Plate Detector
        ↓
Crop License Plate
        ↓
Character Detector
        ↓
Crop Characters
        ↓
CNN Character Classifier
        ↓
License Plate Text
```

This would require a dataset containing:

```text
Vehicle images
+
License plate bounding boxes
```

A YOLO model could then be trained with:

```text
class 0 = license_plate
```

The final system would therefore contain:

```text
YOLO Model #1
→ Detect license plate

YOLO Model #2
→ Detect characters

CNN
→ Classify characters
```

---

## Possible Future Improvements

Future versions of this project may include:

- License plate detection from full vehicle images
- Automatic perspective correction
- Improved handling of rotated plates
- Improved character ordering
- Better handling of Persian RTL/LTR text
- Character detector training for more epochs
- Data augmentation
- Handling multiple license plates in one image
- Real-time video recognition
- Web or mobile interface
- API deployment
- End-to-end license plate recognition service

---

## Technologies Used

- Python
- TensorFlow
- Keras
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Git

---

## Hardware Used During Development

The project was developed and trained on:

```text
Apple MacBook Air M3
16 GB Unified Memory
Apple Silicon GPU
```

YOLO training used:

```text
PyTorch MPS
```

which allows PyTorch workloads to run on the Apple GPU.

---

## Project Status

Current status:

```text
[Completed] XML annotation parsing
[Completed] Character preprocessing
[Completed] CNN character classifier
[Completed] CNN evaluation
[Completed] YOLO dataset generation
[Completed] YOLO character detector
[Completed] Character bounding box sorting
[Completed] Character cropping
[Completed] CNN + YOLO integration
[Completed] End-to-end plate image inference
[Planned] License plate detection from full vehicle images
```

---

## Author

Developed as a deep learning and computer vision learning project for Iranian license plate recognition.