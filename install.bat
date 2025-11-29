@echo off
echo Updating pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Installing dependencies...
python -m pip install flask flask-sqlalchemy flask-login werkzeug
python -m pip install numpy scipy
python -m pip install scikit-learn
python -m pip install pandas
python -m pip install xgboost
python -m pip install librosa soundfile
python -m pip install joblib

echo.
echo Installation complete!
pause

