$ErrorActionPreference = "Stop"
Set-Location "D:\Gesture-Controlled-Virtual-Mouse"
$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Virtual environment not found: $Python" }

& $Python -m pip install --upgrade pip
& $Python -m pip install "vosk==0.3.45" "sounddevice==0.4.6" "pyttsx3==2.90"
if ($LASTEXITCODE -ne 0) { throw "Package installation failed." }

New-Item -ItemType Directory -Force -Path ".\models" | Out-Null
$ModelPath = ".\models\vosk-model-small-en-us-0.15"
if (-not (Test-Path $ModelPath)) {
    $Zip = ".\models\vosk-model-small-en-us-0.15.zip"
    Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile $Zip
    Expand-Archive -Path $Zip -DestinationPath ".\models" -Force
    Remove-Item $Zip -Force
}
Write-Host "Voice packages and Vosk model installed."
