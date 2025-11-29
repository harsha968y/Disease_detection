@echo off
echo ========================================
echo Installing for Python 3.13
echo ========================================
echo.
echo Note: Some packages may need to be installed from source
echo or may require pre-release versions for Python 3.13
echo.

echo Step 1: Updating pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Step 2: Installing numpy and scipy...
python -m pip install numpy scipy

echo.
echo Step 3: Installing scikit-learn (may take longer)...
python -m pip install scikit-learn --no-build-isolation

echo.
echo Step 4: Installing Flask packages...
python -m pip install flask flask-sqlalchemy flask-login werkzeug

echo.
echo Step 5: Installing pandas...
python -m pip install pandas

echo.
echo Step 6: Installing joblib...
python -m pip install joblib

echo.
echo Step 7: Installing xgboost...
python -m pip install xgboost

echo.
echo Step 8: Installing audio libraries...
python -m pip install librosa soundfile

echo.
echo ========================================
echo Installation complete!
echo ========================================
pause

