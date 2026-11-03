@echo off
chcp 65001 >nul
title Court Vision - Phan tich video cau long
set "PY=%USERPROFILE%\court-vision-ai\venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Chua cai dat moi truong AI. Hay chay "Cai dat.bat" truoc.
  pause
  exit /b 1
)
"%PY%" "%~dp0analyzer\launcher.py"
echo.
pause
