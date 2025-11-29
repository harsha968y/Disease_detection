# Installation Guide - Fixing scikit-learn Error

## Problem
The `metadata-generation-failed` error for scikit-learn on Windows is common. Here are solutions:

## Solution 1: Install Packages One by One (Recommended)

Run these commands in order:

```powershell
# 1. Update pip first
python -m pip install --upgrade pip setuptools wheel

# 2. Install numpy and scipy first (dependencies)
python -m pip install numpy scipy

# 3. Install scikit-learn using pre-built wheels only
python -m pip install scikit-learn --only-binary :all:

# 4. Install Flask and related packages
python -m pip install flask flask-sqlalchemy flask-login werkzeug

# 5. Install pandas
python -m pip install pandas

# 6. Install joblib
python -m pip install joblib

# 7. Install xgboost (may take a while)
python -m pip install xgboost

# 8. Install audio processing libraries
python -m pip install librosa soundfile
```

## Solution 2: Use Pre-built Wheels for All Packages

If Solution 1 doesn't work, try installing all packages with pre-built wheels:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary :all: -r requirements.txt
```

## Solution 3: Install Microsoft Visual C++ Build Tools

If you still get errors, you may need to install Microsoft Visual C++ Build Tools:

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++" workload
3. Restart your computer
4. Try installing again

## Solution 4: Use Conda (Alternative)

If pip continues to fail, consider using conda:

```bash
conda install -c conda-forge scikit-learn numpy scipy pandas flask flask-sqlalchemy flask-login werkzeug joblib librosa soundfile
pip install xgboost
```

## Solution 5: Minimal Installation (For Testing)

If you just want to test the app without all features, install minimal dependencies:

```powershell
python -m pip install flask flask-sqlalchemy flask-login werkzeug
python -m pip install numpy scipy scikit-learn --only-binary :all:
python -m pip install pandas joblib
```

Then comment out xgboost and librosa features in the code temporarily.

## Verify Installation

After installation, verify packages are installed:

```powershell
python -c "import sklearn; import flask; import pandas; import numpy; print('All packages installed successfully!')"
```

## Run the Application

Once installed, run:

```powershell
python app.py
```

If you encounter any specific errors, note the error message and we can troubleshoot further.

