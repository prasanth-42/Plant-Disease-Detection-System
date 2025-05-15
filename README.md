# Plant Disease Detection System

## Overview
This is a machine learning project for detecting plant diseases using deep learning. The application uses a pre-trained Keras model to classify plant leaf images across multiple plant types and disease conditions.

## Features
- Supports disease detection for various plants including:
  - Apple
  - Blueberry
  - Cherry
  - Corn
  - Grape
  - Orange
  - Peach
  - Pepper
  - Potato
  - Raspberry
  - Soybean
  - Squash
  - Strawberry
  - Tomato

- Detects multiple disease conditions and healthy states

## Requirements
- Python 3.10+
- Streamlit
- TensorFlow
- Pillow
- NumPy

## Installation
1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application
```
streamlit run app.py
```

## Model
- Model file: `best_model.keras`
- Trained on a dataset of plant leaf images
- Classifies images into specific disease or healthy categories

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


