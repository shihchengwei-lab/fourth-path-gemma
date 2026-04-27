@echo off
set "OLLAMA_MODELS=E:\ollama-models"
setx OLLAMA_MODELS "E:\ollama-models" >nul

tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if errorlevel 1 (
  start "Ollama" /min "E:\Ollama\ollama.exe" serve
)

ping 127.0.0.1 -n 4 >nul
curl.exe -s http://localhost:11434/api/tags
echo.
