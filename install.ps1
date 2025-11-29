Write-Host "Updating pip, setuptools, and wheel..." -ForegroundColor Green
python -m pip install --upgrade pip setuptools wheel

Write-Host "`nInstalling dependencies..." -ForegroundColor Green
python -m pip install flask flask-sqlalchemy flask-login werkzeug
python -m pip install numpy scipy
python -m pip install scikit-learn
python -m pip install pandas
python -m pip install xgboost
python -m pip install librosa soundfile
python -m pip install joblib

Write-Host "`nInstallation complete!" -ForegroundColor Green
Read-Host "Press Enter to continue"

