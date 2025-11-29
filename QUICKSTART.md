# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Train Models (First Time Only)

Train all machine learning models:

```bash
python train_models.py
```

This will:
- Create a sample dataset if the actual UCI Parkinson's dataset is not found
- Train 4 ML models (SVM, KNN, Random Forest, XGBoost)
- Save models in the `models/` directory
- Generate performance metrics

**Note**: For best results, download the actual UCI Parkinson's dataset from:
https://archive.ics.uci.edu/ml/datasets/parkinsons

Save it as `parkinsons_data.csv` in the project root.

## Step 3: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Step 4: Use the Application

1. **Sign Up**: Create a new account
2. **Login**: Use your credentials to login
3. **Dashboard**: 
   - Record a voice sample using the microphone button, OR
   - Upload an audio file (WAV, MP3, etc.)
   - Click "Analyze Voice" to get predictions
4. **View Results**: See predictions from all 4 ML models
5. **View Metrics**: Click "Load Metrics" to see model performance

## Features

- ✅ Multiple ML Models (SVM, KNN, Random Forest, XGBoost)
- ✅ Voice Recording & Upload
- ✅ Real-time Analysis
- ✅ Performance Metrics (Accuracy, Sensitivity, Specificity)
- ✅ User Authentication
- ✅ Modern Web Interface

## Troubleshooting

### Models not found error
Run `python train_models.py` to train the models first.

### Microphone access denied
Allow microphone permissions in your browser settings.

### Audio file not supported
Ensure the audio file is in a supported format (WAV, MP3, FLAC, etc.)

### Low accuracy
Use the actual UCI Parkinson's dataset instead of the synthetic one for better results.

