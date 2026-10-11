# Iranian License Plate Recognition

An end-to-end deep learning and computer vision system for detecting and recognizing Iranian license plates from full vehicle images.

The project uses three trained neural-network models:

1. **YOLO License Plate Detector** — locates the license plate in a full vehicle image.
2. **YOLO Character Detector** — locates individual characters inside the detected plate.
3. **CNN Character Classifier** — recognizes each detected character.

The final pipeline can receive a full image of a vehicle and return the predicted Iranian license plate text.

---

## Project Overview

The system is divided into three main machine learning stages.

### 1. License Plate Detection

The first YOLO model receives a full vehicle image and detects the location of the license plate.

Its job is to answer:

```text
Where is the license plate?
```

The detector uses one object class:

```text
0 = license_plate
```

Its output is a bounding box around the detected plate.

---

### 2. Character Detection

After the license plate is detected, the plate region is cropped from the original image.

The cropped plate is then passed to a second YOLO model.

This model detects the location of individual characters inside the plate.

Its job is to answer:

```text
Where are the characters?
```

The detector uses one object class:

```text
0 = character
```

The detected character bounding boxes are sorted from left to right before classification.

---

### 3. Character Classification

Each detected character is cropped and passed to a Convolutional Neural Network.

The CNN answers:

```text
Which character is this?
```

The classifier was trained on 31 Iranian license plate character classes.

---

# Complete System Pipeline

The complete recognition pipeline is:

```text
Full Vehicle Image
        ↓
YOLO License Plate Detector
        ↓
License Plate Bounding Box
        ↓
Crop License Plate
        ↓
YOLO Character Detector
        ↓
Character Bounding Boxes
        ↓
Sort Characters Left-to-Right
        ↓
Crop Individual Characters
        ↓
Resize to 32 × 64
        ↓
Convert to Grayscale
        ↓
Normalize Pixel Values
        ↓
CNN Character Classifier
        ↓
Predicted Character Labels
        ↓
Final License Plate Text
```

The system therefore contains:

```text
YOLO Model #1
→ License Plate Detection

YOLO Model #2
→ Character Detection

CNN Model
→ Character Classification
```

---

# Project Structure

```text
iranian-license-plate-ocr/
│
├── data/
│   ├── train/
│   ├── validation/
│   ├── test/
│   │
│   ├── yolo_dataset/
│   │
│   ├── plate_train/
│   ├── plate_validation/
│   ├── plate_test/
│   │
│   └── plate_yolo_dataset/
│
├── models/
│   ├── license_plate_character_cnn.keras
│   ├── license_plate_character_detector.pt
│   ├── license_plate_detector.pt
│   └── class_mapping.json
│
├── src/
│   ├── read_annotation.ipynb
│   ├── train_character_detector.ipynb
│   ├── train_plate_detector.ipynb
│   └── Inference.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

The large datasets and YOLO training run directories are excluded from Git using `.gitignore`.

---

# Notebook Description

## `read_annotation.ipynb`

This notebook contains the CNN character classification pipeline.

Main tasks:

- Parse XML annotation files
- Read character labels
- Extract character bounding boxes
- Validate bounding box coordinates
- Crop individual characters
- Resize characters to `32 × 64`
- Convert images to grayscale
- Normalize pixel values
- Encode character labels
- Prepare train, validation, and test datasets
- Build the CNN architecture
- Train the CNN classifier
- Evaluate the classifier
- Analyze incorrect predictions
- Generate confusion matrices
- Save the trained CNN model
- Save the character class mapping

The trained CNN is stored as:

```text
models/license_plate_character_cnn.keras
```

The class mapping is stored as:

```text
models/class_mapping.json
```

---

## `train_character_detector.ipynb`

This notebook contains the YOLO character detection pipeline.

Main tasks:

- Read character bounding boxes from XML annotations
- Convert XML bounding boxes to YOLO format
- Generate YOLO annotation files
- Build the YOLO character dataset
- Configure the YOLO dataset
- Load a pretrained YOLO11n model
- Fine-tune YOLO for character detection
- Evaluate the detector
- Test character detection
- Sort character bounding boxes
- Crop detected characters
- Connect YOLO detections to the CNN classifier

The final trained detector is stored as:

```text
models/license_plate_character_detector.pt
```

The detector uses one class:

```text
0 = character
```

---

## `train_plate_detector.ipynb`

This notebook contains the license plate detection pipeline.

Its purpose is to detect the entire license plate from a full vehicle image.

Main tasks:

- Read vehicle image XML annotations
- Extract the annotation labeled as the full plate region
- Visualize license plate bounding boxes
- Convert plate annotations to YOLO format
- Validate and clip bounding box coordinates
- Build the YOLO plate detection dataset
- Create train, validation, and test splits
- Configure the plate detection dataset
- Load a pretrained YOLO11n model
- Fine-tune YOLO for license plate detection
- Evaluate the trained plate detector
- Detect plates in full vehicle images
- Crop detected plates
- Connect the plate detector to the OCR pipeline

The plate detector uses one class:

```text
0 = license_plate
```

The final trained detector should be stored as:

```text
models/license_plate_detector.pt
```

---

## `Inference.ipynb`

This notebook contains the final inference pipeline.

It loads all three trained models:

```text
YOLO License Plate Detector
+
YOLO Character Detector
+
CNN Character Classifier
```

It also loads:

```text
Character Class Mapping
```

The notebook can process a full vehicle image and perform:

```text
Vehicle Image
↓
License Plate Detection
↓
Plate Cropping
↓
Character Detection
↓
Character Sorting
↓
Character Cropping
↓
CNN Preprocessing
↓
Character Classification
↓
Final Plate Text
```

This notebook is the main entry point for using the trained system.

---

# CNN Character Classifier

The character classifier is implemented using TensorFlow / Keras.

## CNN Architecture

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

# CNN Performance

The CNN was evaluated on the test dataset.

Final test results:

```text
Test Accuracy: 97.38%
Test Loss: 0.117
```

A confusion matrix was also generated to analyze the most common classification errors.

---

# YOLO Character Detector

The character detector is based on:

```text
YOLO11n
```

A pretrained YOLO11n model was fine-tuned using character-level bounding boxes extracted from the original XML annotations.

Training configuration used during the initial experiment:

```text
Epochs: 3
Image Size: 640
Batch Size: 16
Device: Apple MPS
```

The model was trained using the Apple Silicon GPU through PyTorch MPS.

---

# Character Detector Validation Results

Final validation results from the trained character detector:

```text
Precision: 0.973
Recall: 0.987
mAP50: 0.989
mAP50-95: 0.684
```

The detector successfully locates individual characters inside Iranian license plates with high precision and recall.

---

# License Plate Detector

A second YOLO11n model is used to locate the license plate itself from a full vehicle image.

Unlike the character detector, this model receives images containing the entire vehicle.

Its task is:

```text
Full Vehicle Image
↓
Detect License Plate
```

The dataset contains an XML object representing the complete plate region:

```text
کل ناحیه پلاک
```

Only this object is used when training the plate detector.

Other character annotations in the same XML files are ignored during plate-level detection training.

---

# Plate Detection Annotation Format

The source dataset stores bounding boxes as:

```text
xmin
ymin
xmax
ymax
```

For YOLO training, they are converted to:

```text
class_id
x_center
y_center
width
height
```

All YOLO coordinates are normalized relative to the image dimensions.

Example:

```text
0 0.41 0.55 0.22 0.08
```

Where:

```text
0       = license_plate class
0.41    = normalized x center
0.55    = normalized y center
0.22    = normalized width
0.08    = normalized height
```

Bounding boxes are also clipped to the valid image boundaries before generating YOLO labels.

---

# Character Detection Annotation Format

Character annotations are converted to the same YOLO representation:

```text
0 x_center y_center width height
```

Where:

```text
0 = character
```

The character detector does not classify the character.

It only determines where a character exists.

Character identity is determined later by the CNN classifier.

---

# Character Preprocessing

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
Normalize Pixel Values to 0–1
↓
Add Channel Dimension
```

The final shape of one character is:

```text
(64, 32, 1)
```

For multiple characters:

```text
(number_of_characters, 64, 32, 1)
```

---

# Plate Recognition Function

The complete pipeline can be represented conceptually as:

```python
recognize_plate_from_vehicle(image_path)
```

The function:

1. Loads the full vehicle image
2. Detects the license plate
3. Selects the detected plate
4. Crops the plate
5. Detects individual characters
6. Sorts characters from left to right
7. Crops each character
8. Preprocesses character images
9. Passes them to the CNN
10. Converts CNN predictions to labels
11. Builds the final plate text

Example conceptual result:

```python
{
    "plate_text": "...",
    "labels": [...],
    "plate_crop": ...,
    "plate_confidence": ...
}
```

---

# Installation

Create a Python virtual environment.

On macOS/Linux:

```bash
python3 -m venv ai1_env
```

Activate it:

```bash
source ai1_env/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# Main Libraries

The main technologies and libraries used in this project are:

```text
Python
TensorFlow
Keras
PyTorch
Ultralytics YOLO
OpenCV
NumPy
Matplotlib
Scikit-learn
ElementTree
Jupyter Notebook
Git
```

---

# Running the Project

Open:

```text
src/Inference.ipynb
```

Make sure the trained models exist:

```text
models/license_plate_detector.pt
models/license_plate_character_detector.pt
models/license_plate_character_cnn.keras
models/class_mapping.json
```

Provide the path to a full vehicle image.

Example:

```python
image_path = "/path/to/vehicle.jpg"
```

Run the full recognition pipeline:

```python
result = recognize_plate_from_vehicle(image_path)
```

Example:

```python
if result is None:
    print("No plate detected.")
else:
    print("Predicted Plate:", result["plate_text"])
    print("Plate Confidence:", result["plate_confidence"])
```

---

# Recognition Example

The system performs:

```text
Vehicle Image
↓
Plate Detected
↓
Plate Cropped
↓
Characters Detected
↓
Characters Classified
↓
Plate Text Returned
```

Example predicted character list:

```python
['1', '3', 'ب', '1', '1', '1', '1', '0']
```

Because Persian characters use right-to-left text direction while digits are rendered left-to-right, mixed Persian and numeric plate text may require Unicode bidirectional handling when displayed.

---

# Datasets

The project currently uses two related datasets.

## Character Recognition Dataset

Used for:

```text
Character Detection
+
Character Classification
```

Structure:

```text
data/
├── train/
├── validation/
└── test/
```

Each image has a corresponding XML annotation file containing individual character labels and bounding boxes.

---

## License Plate Detection Dataset

Used for:

```text
Full Vehicle Image
→ License Plate Detection
```

Structure:

```text
data/
├── plate_train/
├── plate_validation/
└── plate_test/
```

Each XML annotation contains:

```text
Full plate bounding box
+
Individual character annotations
```

For the plate detector, only the object representing:

```text
کل ناحیه پلاک
```

is used.

---

# Generated YOLO Datasets

The project generates separate YOLO datasets for the two detection tasks.

Character detection:

```text
data/yolo_dataset/
```

Plate detection:

```text
data/plate_yolo_dataset/
```

Typical structure:

```text
images/
├── train/
├── validation/
└── test/

labels/
├── train/
├── validation/
└── test/

data.yaml
```

The YAML configuration tells YOLO:

```text
Where the dataset is located
Where train data is located
Where validation data is located
Where test data is located
Which object classes exist
```

---

# Dataset and Annotation Issues

Some source annotations contain invalid or unusual coordinates.

Examples encountered during development include:

```text
Coordinates outside image boundaries
Floating-point coordinates such as 435.0
Very small images
Invalid YOLO-normalized coordinates
```

The preprocessing pipeline handles these cases by:

```text
Parsing coordinates as numeric values
Clipping bounding boxes to image boundaries
Skipping invalid bounding boxes
```

Some corrupted samples may also be ignored automatically by Ultralytics during training.

---

# Current Capabilities

The current project supports:

```text
[Completed] XML annotation parsing

[Completed] Character preprocessing

[Completed] CNN character classifier

[Completed] CNN test evaluation

[Completed] Character confusion matrix analysis

[Completed] YOLO character dataset generation

[Completed] YOLO character detector

[Completed] Character bounding box sorting

[Completed] Character cropping

[Completed] CNN + Character YOLO integration

[Completed] License plate detection dataset preparation

[Completed] YOLO license plate detector

[Completed] License plate cropping from full vehicle images

[Completed] Plate Detector + Character Detector + CNN integration

[Completed] End-to-end full vehicle image recognition
```

---

# Current System Architecture

```text
                  Vehicle Image
                       ↓
             License Plate Detector
                       ↓
                  Plate Crop
                       ↓
              Character Detector
                       ↓
                Character Crops
                       ↓
                CNN Classifier
                       ↓
                License Plate Text
```

---

# Planned Real-World Architecture

A future deployment may use the recognition system as an edge AI component.

Example:

```text
Camera
↓
Raspberry Pi AI Device
↓
Plate Recognition Pipeline
↓
Recognized Plate
↓
HTTP / REST API
↓
Node.js Backend
↓
Authorization / Database
↓
Gate Control
```

The AI device should perform image processing and recognition.

The backend should remain responsible for:

```text
Authorization
Database access
User permissions
Event logging
Gate control
Smart-home business logic
```

This keeps the AI recognition component separate from the main system backend.

---

# Planned Live Camera Support

The image-based pipeline can later be connected to an OpenCV camera stream.

Conceptually:

```text
Camera
↓
Video Frame
↓
Plate Recognition Pipeline
↓
Predicted Plate
```

Instead of passing an image path, OpenCV frames can be passed directly to the trained models.

For a real deployment, additional features will be useful:

```text
Frame skipping
Duplicate plate suppression
Tracking
Confidence filtering
Request retry handling
Local logging
Backend communication
```

---

# Possible Future Improvements

Possible future improvements include:

- Real-time camera recognition
- Raspberry Pi deployment
- Node.js backend integration
- REST API communication
- Duplicate plate detection suppression
- Vehicle tracking
- License plate super-resolution
- Image enhancement for low-quality plate crops
- Sharpening and denoising
- Improved handling of rotated plates
- Perspective correction
- More plate detector training epochs
- Data augmentation
- Improved low-light recognition
- Multiple vehicle / multiple plate detection
- Vehicle make and model recognition
- Vehicle color recognition
- Parking management integration
- Smart gate integration
- API service deployment

---

# Plate Image Enhancement

Low-resolution and distant license plates may produce low-quality plate crops.

A possible future stage could be added between plate detection and character detection:

```text
Plate Detection
↓
Plate Crop
↓
Image Enhancement / Super Resolution
↓
Character Detection
↓
CNN Classification
```

Possible techniques include:

```text
OpenCV sharpening
Denoising
Super-resolution
Real-ESRGAN
Dedicated plate enhancement models
```

This feature is not required for the current recognition pipeline.

---

# Vehicle Information Recognition

Another possible extension is vehicle attribute recognition.

Future architecture:

```text
Vehicle Image
├── License Plate Recognition
├── Vehicle Make / Model Recognition
└── Vehicle Color Recognition
```

These tasks would require separate datasets and models.

---

# Hardware Used During Development

Development and model training were performed on:

```text
Apple MacBook Air M3
16 GB Unified Memory
Apple Silicon GPU
```

YOLO training used:

```text
PyTorch MPS
```

which allows compatible PyTorch workloads to run on the Apple GPU.

---

# Git and Generated Files

Large datasets and YOLO experiment outputs should not normally be committed to Git.

Examples:

```text
data/
src/runs/
```

The final trained models should be stored in:

```text
models/
```

This keeps the repository focused on:

```text
Source code
Jupyter notebooks
Model artifacts
Documentation
Configuration
```

---

# Project Status

Current project status:

```text
[Completed] Character classification model
[Completed] Character detection model
[Completed] License plate detection model
[Completed] Full vehicle image input
[Completed] License plate cropping
[Completed] End-to-end OCR pipeline

[Planned] Live camera deployment
[Planned] Raspberry Pi deployment
[Planned] Backend integration
[Planned] Vehicle color recognition
[Planned] Vehicle make/model recognition
[Planned] Plate image enhancement
```

---

# Author

Developed as a deep learning and computer vision project for Iranian license plate detection and recognition.

The project was built incrementally to study and implement:

- Image preprocessing
- CNN classification
- Object detection
- YOLO
- XML annotation processing
- Model evaluation
- Multi-model inference pipelines
- Computer vision system integration