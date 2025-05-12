# Signature Feature Extraction and Processing

This project provides tools for processing handwritten signature images by cropping, resizing, and extracting distinguishing features using image processing and computer vision techniques.

## Features

- Automatically crops and resizes signature images to a standard size.
- Extracts numerical features from processed signature images.
- Saves features in JSON format for training or analysis.
- Supports both original and forged signature directories.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Krish2208/Forgery-Detection-PR 
# or
git clone git@github.com:Krish2208/Forgery-Detection-PR.git

cd Forgery-Detection-PR

# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## Usage
To process the signatures and extract features, run:

```bash
python process_signatures.py
```

Once the above step is done, run:

```bash
streamlit run interface.py
```