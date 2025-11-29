import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import json

# Feature names for Parkinson's dataset (31 features)
FEATURE_NAMES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
]

def extract_voice_features(audio_path):
    """
    Extract features from voice audio file.
    Returns a feature vector compatible with the Parkinson's dataset.
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract various audio features
        features = []
        
        # Fundamental frequency (F0) features
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0:
            return None
        
        # Mean, max, min F0
        mean_f0 = np.mean(f0_clean)
        max_f0 = np.max(f0_clean)
        min_f0 = np.min(f0_clean)
        
        features.extend([mean_f0, max_f0, min_f0])
        
        # Jitter (variation in F0)
        if len(f0_clean) > 1:
            jitter = np.mean(np.abs(np.diff(f0_clean))) / mean_f0
            jitter_abs = np.mean(np.abs(np.diff(f0_clean)))
            rap = np.mean(np.abs(np.diff(f0_clean))) / mean_f0
            ppq = np.percentile(np.abs(np.diff(f0_clean)), 50) / mean_f0
            ddp = 3 * jitter
        else:
            jitter = jitter_abs = rap = ppq = ddp = 0.0
        
        features.extend([jitter, jitter_abs, rap, ppq, ddp])
        
        # Shimmer (amplitude variation)
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        if len(rms) > 1:
            shimmer = np.mean(np.abs(np.diff(rms))) / np.mean(rms)
            shimmer_db = 20 * np.log10(np.max(rms) / np.min(rms)) if np.min(rms) > 0 else 0
            apq3 = np.mean(np.abs(np.diff(rms[:len(rms)//3]))) / np.mean(rms[:len(rms)//3]) if len(rms) >= 3 else 0
            apq5 = np.mean(np.abs(np.diff(rms[:len(rms)//5]))) / np.mean(rms[:len(rms)//5]) if len(rms) >= 5 else 0
            apq = np.mean(np.abs(np.diff(rms))) / np.mean(rms)
            dda = 3 * shimmer
        else:
            shimmer = shimmer_db = apq3 = apq5 = apq = dda = 0.0
        
        features.extend([shimmer, shimmer_db, apq3, apq5, apq, dda])
        
        # Harmonic-to-noise ratio (HNR)
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = np.mean(harmonic) / (np.mean(percussive) + 1e-10)
        nhr = 1 / (hnr + 1e-10)
        
        features.extend([nhr, hnr])
        
        # Nonlinear features
        # RPDE (Recurrence Period Density Entropy)
        rpde = calculate_rpde(f0_clean)
        features.append(rpde)
        
        # DFA (Detrended Fluctuation Analysis)
        dfa = calculate_dfa(f0_clean)
        features.append(dfa)
        
        # Spread features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spread1 = np.std(mfccs[0])
        spread2 = np.std(mfccs[1])
        features.extend([spread1, spread2])
        
        # D2 (Correlation dimension)
        d2 = calculate_d2(f0_clean)
        features.append(d2)
        
        # PPE (Pitch Period Entropy)
        ppe = calculate_ppe(f0_clean)
        features.append(ppe)
        
        # Ensure we have exactly 22 features (matching the dataset)
        if len(features) < 22:
            features.extend([0.0] * (22 - len(features)))
        elif len(features) > 22:
            features = features[:22]
        
        return np.array(features)
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def calculate_rpde(signal):
    """Calculate Recurrence Period Density Entropy"""
    try:
        if len(signal) < 10:
            return 0.0
        # Simplified RPDE calculation
        diff_signal = np.diff(signal)
        hist, _ = np.histogram(diff_signal, bins=10)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        return entropy
    except:
        return 0.0

def calculate_dfa(signal):
    """Calculate Detrended Fluctuation Analysis"""
    try:
        if len(signal) < 10:
            return 0.0
        # Simplified DFA calculation
        n = len(signal)
        y = np.cumsum(signal - np.mean(signal))
        scales = np.logspace(1, np.log10(n//4), 10).astype(int)
        fluctuations = []
        for scale in scales:
            if scale >= n:
                continue
            segments = n // scale
            f = []
            for i in range(segments):
                segment = y[i*scale:(i+1)*scale]
                if len(segment) > 0:
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    f.append(np.std(segment - trend))
            if len(f) > 0:
                fluctuations.append(np.mean(f))
        if len(fluctuations) > 0:
            return np.mean(fluctuations)
        return 0.0
    except:
        return 0.0

def calculate_d2(signal):
    """Calculate Correlation Dimension D2"""
    try:
        if len(signal) < 10:
            return 0.0
        # Simplified D2 calculation
        diff_signal = np.diff(signal)
        return np.std(diff_signal) / (np.mean(np.abs(diff_signal)) + 1e-10)
    except:
        return 0.0

def calculate_ppe(signal):
    """Calculate Pitch Period Entropy"""
    try:
        if len(signal) < 10:
            return 0.0
        periods = np.diff(signal)
        hist, _ = np.histogram(periods, bins=10)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        return entropy
    except:
        return 0.0

def load_or_create_dataset():
    """Load Parkinson's dataset or create a sample one"""
    dataset_path = 'parkinsons_data.csv'
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        # Handle different column name formats
        if 'status' not in df.columns:
            # Check for common alternative names
            if 'Status' in df.columns:
                df['status'] = df['Status']
            elif 'class' in df.columns:
                df['status'] = df['class']
            elif 'target' in df.columns:
                df['status'] = df['target']
        
        # Ensure name column exists
        if 'name' not in df.columns:
            if 'Name' in df.columns:
                df['name'] = df['Name']
            else:
                df['name'] = [f'patient_{i+1}' for i in range(len(df))]
        
        return df
    
    # Create sample dataset structure
    # This is a placeholder - in production, use the actual UCI Parkinson's dataset
    print("Dataset not found. Please download the Parkinson's dataset from UCI ML Repository.")
    print("Creating sample dataset structure...")
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 195
    n_features = 22
    
    # Healthy samples (172)
    healthy_data = np.random.randn(172, n_features)
    healthy_data = healthy_data * 0.5 + np.random.randn(1, n_features) * 0.3
    
    # Parkinson's samples (23)
    parkinsons_data = np.random.randn(23, n_features)
    parkinsons_data = parkinsons_data * 0.8 + np.random.randn(1, n_features) * 0.5
    # Shift some features to simulate Parkinson's characteristics
    parkinsons_data[:, 0] += 0.5  # F0 shift
    parkinsons_data[:, 3] += 0.3  # Jitter increase
    parkinsons_data[:, 8] += 0.2  # Shimmer increase
    
    # Combine data
    X = np.vstack([healthy_data, parkinsons_data])
    y = np.hstack([np.zeros(172), np.ones(23)])
    
    # Create DataFrame
    feature_names = FEATURE_NAMES[:n_features]
    df = pd.DataFrame(X, columns=feature_names)
    df['status'] = y.astype(int)
    df['name'] = [f'patient_{i+1}' for i in range(n_samples)]
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df.to_csv(dataset_path, index=False)
    print(f"Sample dataset created with {n_samples} samples")
    
    return df

def train_all_models():
    """Train all ML models and save them"""
    print("Loading dataset...")
    df = load_or_create_dataset()
    
    # Prepare data - drop non-feature columns
    columns_to_drop = ['status', 'name', 'Status', 'Name', 'class', 'target']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    X = df.drop(columns_to_drop, axis=1).values
    y = df['status'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize models
    models = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    metrics = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Sensitivity (Recall) and Specificity
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics[name] = {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            'f1_score': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0
        }
        
        # Save model
        model_path = f'models/{name.lower().replace(" ", "_")}_model.pkl'
        joblib.dump(model, model_path)
        print(f"{name} - Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    
    # Save metrics
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def load_trained_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'SVM': 'models/svm_model.pkl',
        'KNN': 'models/knn_model.pkl',
        'Random Forest': 'models/random_forest_model.pkl',
        'XGBoost': 'models/xgboost_model.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            print(f"Warning: {name} model not found. Please train models first.")
    
    return models

def predict_parkinsons(features, model_name='Random Forest'):
    """Predict Parkinson's disease using specified model"""
    models = load_trained_models()
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found")
    
    model = models[model_name]
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    return {
        'prediction': 'Parkinson\'s Disease' if prediction == 1 else 'Healthy',
        'probability': float(probability[1] if prediction == 1 else probability[0]),
        'confidence': float(max(probability)) * 100
    }

