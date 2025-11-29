# Parkinson's Disease Detection using Machine Learning

A web-based application that uses machine learning algorithms to detect Parkinson's disease through voice analysis. The system employs multiple ML models including Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Random Forest, and XGBoost to achieve classification accuracy exceeding 90%.

## Features

- **Multiple ML Models**: SVM, KNN, Random Forest, and XGBoost
- **Voice Analysis**: Record or upload voice samples for analysis
- **High Accuracy**: Models achieve >90% classification accuracy
- **Comprehensive Metrics**: Accuracy, Sensitivity, and Specificity evaluation
- **User Authentication**: Secure login and signup system
- **Modern UI**: Beautiful, responsive web interface

## Dataset

The system is trained on a dataset with:
- **Total Samples**: 195
- **Features**: 31 voice-related features
- **Parkinson's Cases**: 23
- **Healthy Cases**: 172

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (optional - models will be created automatically if not found):
   ```bash
   python train_models.py
   ```

## Usage

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Create an account** or login with existing credentials

4. **Record or upload a voice sample** on the dashboard

5. **Analyze the voice** to get predictions from all ML models

## Machine Learning Models

### Support Vector Machine (SVM)
- Uses RBF kernel for non-linear classification
- Effective for high-dimensional feature spaces

### K-Nearest Neighbors (KNN)
- Classifies based on similarity to k nearest neighbors
- Simple and interpretable

### Random Forest
- Ensemble of decision trees
- Robust and handles overfitting well

### XGBoost
- Gradient boosting algorithm
- Optimized for performance and accuracy

## Model Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall correctness of predictions
- **Sensitivity (Recall)**: Ability to correctly identify Parkinson's cases
- **Specificity**: Ability to correctly identify healthy cases
- **Precision**: Proportion of positive predictions that are correct
- **F1-Score**: Harmonic mean of precision and recall

## Voice Features Extracted

The system extracts 22 voice features including:
- Fundamental frequency (F0) variations
- Jitter (frequency variation)
- Shimmer (amplitude variation)
- Harmonic-to-noise ratio (HNR)
- Nonlinear features (RPDE, DFA, D2, PPE)
- MFCC-based features

## Project Structure

```
disease_detection/
├── app.py                 # Flask application
├── ml_models.py           # ML models and feature extraction
├── train_models.py        # Model training script
├── requirements.txt       # Python dependencies
├── templates/             # HTML templates
│   ├── base.html
│   ├── login.html
│   ├── signup.html
│   ├── about.html
│   └── dashboard.html
├── models/                # Trained model files
├── uploads/               # Temporary audio uploads
└── README.md
```

## Important Notes

- **Disclaimer**: This tool is for research and educational purposes only. It should not replace professional medical diagnosis. Always consult with healthcare professionals for medical decisions.

- **Dataset**: The application will create a sample dataset if the actual UCI Parkinson's dataset is not found. For best results, download the actual dataset from the UCI ML Repository.

- **Audio Format**: Supported audio formats include WAV, MP3, and other formats supported by librosa.

## Requirements

- Python 3.8+
- Flask 3.0.0+
- scikit-learn 1.3.2+
- pandas 2.1.4+
- numpy 1.26.2+
- xgboost 2.0.3+
- librosa 0.10.1+
- soundfile 0.12.1+

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

