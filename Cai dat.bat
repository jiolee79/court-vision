@echo off
chcp 65001 >nul
title Court Vision - Cai dat moi truong AI
rem Cai moi truong Python rieng (ngoai Dropbox) va cac thu vien AI. Chi can chay 1 lan.
set "BASE=%USERPROFILE%\court-vision-ai"
set "PY=%BASE%\venv\Scripts\python.exe"
if not exist "%PY%" python -m venv "%BASE%\venv" || (echo Can cai Python 3.11+ tu python.org & pause & exit /b 1)
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
"%PY%" -m pip install ultralytics openvino nncf openpyxl pandas scipy tqdm gdown pillow
if not exist "%BASE%\models\tracknet\TrackNet_best.pt" (
  echo Dang tai trong so TrackNetV3 ^(MIT License^)...
  mkdir "%BASE%\models\tracknet" 2>nul
  "%PY%" -m gdown "https://drive.google.com/uc?id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA" -O "%BASE%\models\tracknet\ckpts.zip"
  "%PY%" -c "import zipfile,pathlib,shutil;d=pathlib.Path(r'%BASE%\models\tracknet');z=zipfile.ZipFile(d/'ckpts.zip');[(d/pathlib.Path(n).name).write_bytes(z.read(n)) for n in z.namelist() if n.endswith('.pt')];z.close();(d/'ckpts.zip').unlink()"
)
echo.
echo Cai dat xong. Bam dup "Phan tich video.bat" de bat dau.
pause
