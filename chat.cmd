@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "OLLAMA_MODELS=E:\ollama-models"
setx OLLAMA_MODELS "E:\ollama-models" >nul

tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if errorlevel 1 (
  start "Ollama" /min "E:\Ollama\ollama.exe" serve
)

ping 127.0.0.1 -n 4 >nul
python "%~dp0main.py" chat --timeout 900 %*
