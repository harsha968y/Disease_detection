# Fix Installation Issues - Python 3.13

## Issue
You're using Python 3.13.2, which is very new. Some packages (especially scikit-learn) may not have full support yet.

## Quick Fix - Try This First

Run this command to install scikit-learn with pre-release support:

```powershell
python -m pip install --pre scikit-learn
```

## Alternative Solutions

### Option 1: Use Python 3.11 or 3.12 (Recommended)

Python 3.11 or 3.12 have better package support. You can:

1. Install Python 3.11 or 3.12 from python.org
2. Create a virtual environment with that version:
   ```powershell
   py -3.11 -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Option 2: Install All Packages Manually

Run these commands one by one:

```powershell
# Update pip
python -m pip install --upgrade pip setuptools wheel

# Core scientific libraries
python -m pip install numpy scipy

# Try scikit-learn with different flags
python -m pip install --pre scikit-learn
# OR
python -m pip install scikit-learn --no-build-isolation
# OR if above fails, try:
python -m pip install scikit-learn --no-cache-dir

# Flask packages
python -m pip install flask flask-sqlalchemy flask-login werkzeug

# Data processing
python -m pip install pandas joblib

# ML library
python -m pip install xgboost

# Audio processing
python -m pip install librosa soundfile
```

### Option 3: Minimal Version (Skip XGBoost Temporarily)

If xgboost fails, you can modify the code to work without it:

1. Edit `ml_models.py` and comment out XGBoost imports
2. Edit `app.py` to handle missing XGBoost gracefully

### Option 4: Use Conda

Conda often has better package management:

```bash
conda create -n parkinsons python=3.11
conda activate parkinsons
conda install -c conda-forge scikit-learn numpy scipy pandas flask flask-sqlalchemy flask-login werkzeug librosa soundfile joblib
pip install xgboost
```

## Verify Installation

After installation, test:

```powershell
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import flask; print('Flask installed')"
python -c "import pandas; print('Pandas installed')"
python -c "import xgboost; print('XGBoost installed')"
```

## Run the Application

Once everything is installed:

```powershell
python app.py
```

The app should start on http://localhost:5000

## If Still Having Issues

1. Check Python version: `python --version`
2. Check pip version: `python -m pip --version`
3. Try installing in a fresh virtual environment
4. Consider using Python 3.11 or 3.12 for better compatibility

