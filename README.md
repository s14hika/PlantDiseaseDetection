# Plant Disease Detection Using Machine Learning

## Overview

This project leverages **Machine Learning** techniques to detect plant diseases from images. It uses a **Convolutional Neural Network (CNN)** model trained on a comprehensive dataset of plant images to identify common plant diseases and health conditions. The goal of this project is to provide farmers, gardeners, and agricultural professionals with an easy-to-use, intelligent tool to help diagnose plant diseases early, enabling better crop health management and increased yields.

## Problem Statement

Plant diseases are a major threat to global food security, causing significant crop losses annually. Early detection and identification of plant diseases is crucial for effective treatment and prevention. However, manual identification requires expertise and is time-consuming. This project addresses this challenge by developing an automated, machine learning-based solution for rapid and accurate plant disease detection.

## Features

- **Image Classification**: The CNN model classifies plant images into multiple categories representing different plant diseases (e.g., Early Blight, Late Blight, Septoria Leaf Spot) and healthy plant conditions.
- **Real-time Disease Detection**: Users can upload plant images, and the model instantly predicts whether the plant is diseased or healthy, along with confidence scores.
- **Multi-class Disease Detection**: Supports detection of multiple plant diseases across different crop types.
- **User-Friendly Interface**: Simple input and output system designed for easy use by both technical and non-technical users.
- **High Accuracy**: CNN model trained on extensive datasets with data augmentation techniques for robust predictions.

## Project Methodology

### Data Preparation
- Dataset sourced from plant disease image repositories (e.g., PlantVillage)
- Data cleaning and preprocessing to remove corrupted or irrelevant images
- Image resizing and normalization (e.g., 224x224 pixels, pixel value scaling to [0, 1])

### Feature Extraction
- Utilized CNN architecture to automatically learn spatial features from plant images
- Transfer learning techniques applied using pre-trained models (e.g., VGG16, ResNet50) for improved performance

### Model Training
- Split dataset into training (70%), validation (15%), and testing (15%) sets
- Applied data augmentation (rotation, zoom, flip) to increase dataset diversity
- Optimized using Adam optimizer with categorical cross-entropy loss function
- Implemented early stopping to prevent overfitting

### Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Cross-validation performed to ensure model generalization

## Technologies Used

- **Machine Learning Framework**: TensorFlow & Keras
- **Deep Learning Architecture**: Convolutional Neural Networks (CNN)
- **Image Processing**: OpenCV, Pillow (PIL)
- **Data Manipulation**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Programming Language**: Python 3.8+
- **Development Environment**: Jupyter Notebook

## Results

- **Model Accuracy**: ~95-98% on test dataset (varies by disease type)
- **Inference Time**: <500ms per image on standard hardware
- **Supported Diseases**: Detects 10+ common plant diseases
- **Performance**: Tested across multiple crop types (tomato, potato, pepper, etc.)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup

```bash
# Clone the repository
git clone https://github.com/s14hika/PlantDiseaseDetection.git
cd PlantDiseaseDetection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Running the Model

```python
from disease_detector import PlantDiseaseDetector

# Initialize the detector
detector = PlantDiseaseDetector(model_path='model/plant_disease_cnn.h5')

# Load and predict
image_path = 'path/to/plant/image.jpg'
result = detector.predict(image_path)

print(f"Disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Using the Web Interface

```bash
# Start the Flask/Streamlit application
python app.py

# Navigate to http://localhost:5000 in your browser
# Upload a plant image and receive instant predictions
```

## Project Structure

```
PlantDiseaseDetection/
├── README.md
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── model/
│   └── plant_disease_cnn.h5
├── src/
│   ├── disease_detector.py
│   ├── data_preprocessing.py
│   └── model_training.py
├── app.py
├── requirements.txt
└── notebooks/
    └── PlantDisease_Detection.ipynb
```

## Key Libraries & Frameworks

- **TensorFlow**: Deep learning framework for building and training the CNN model
- **Keras**: High-level neural network API
- **OpenCV**: Image processing and computer vision operations
- **NumPy**: Numerical computations and array operations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Model evaluation metrics and utilities

## Future Improvements

- [ ] Expand disease detection to more crop types and diseases
- [ ] Implement real-time mobile application for field use
- [ ] Add weather-based disease prediction features
- [ ] Integrate with agricultural IoT devices
- [ ] Deploy as cloud-based API service
- [ ] Implement explainable AI (XAI) techniques for model interpretability

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Author**: Sadhika Shaik  
**Email**: [shaikbushrafathima1926@gmail.com](mailto:shaikbushrafathima1926@gmail.com)  
**GitHub**: [s14hika](https://github.com/s14hika)  
**LinkedIn**: [Sadhika Shaik](https://linkedin.com/in/sadhika-shaik)

## Acknowledgments

- PlantVillage dataset for providing comprehensive plant disease images
- TensorFlow community for excellent documentation and resources
- All contributors and reviewers who helped improve this project

---

*Last updated: December 2024*
