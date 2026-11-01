# Setup and How to Run Guide

This guide walks you through setting up and running the **Gesture-Controlled Virtual Mouse and Voice Assistant** on a new Windows computer.

---

## 1. System Requirements

* **Operating System**: Windows 10 or 11 (64-bit)
* **Hardware**:
  * Working **Webcam** (built-in or USB)
  * Working **Microphone** (for voice commands)
* **Software**:
  * **Python 3.11** (Recommended) or Python 3.12 ([Download from python.org](https://www.python.org/downloads/))  
    *(Important: Check the box **"Add python.exe to PATH"** during installation)*
  * **Git** ([Download Git for Windows](https://git-scm.com/))

---

## 2. Installation & Setup (One-Time)

Open **PowerShell** or **Command Prompt** and follow these steps:

### Step 1: Clone the Repository
```powershell
git clone https://github.com/PKgamer1358/Virtual_Mouse.git
cd Virtual_Mouse
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv .venv
```

### Step 3: Activate the Environment
* In **PowerShell**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  > *Note: If PowerShell displays a script execution policy error, run:*  
  > `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`  
  > *and re-run the activation command.*

* In **Command Prompt (CMD)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

### Step 4: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip
```

**If you are using Python 3.11 (Recommended):**
```powershell
pip install -r requirements-py311.txt
pip install sounddevice vosk assemblyai
```

**If you are using Python 3.12:**
```powershell
pip install opencv-python mediapipe pyautogui pynput comtypes pycaw screen-brightness-control eel SpeechRecognition pyttsx3 wikipedia sounddevice vosk
```

### Step 5 (Optional): Set Up Offline Voice Model
To download the offline Vosk speech recognition model:
```powershell
powershell -ExecutionPolicy Bypass -File setup_voice.ps1
```

---

## 3. How to Run

Before running any script, make sure your virtual environment is active (`( .venv )` should show in your terminal prompt):
```powershell
.\.venv\Scripts\Activate.ps1
```

### Mode 1: Hand Gesture Virtual Mouse (Recommended)
Controls your cursor, clicks, scrolling, drag-and-drop, volume, and brightness using camera gestures.

```powershell
cd src
python Gesture_Controller.py
```

* **Keyboard Controls in Camera Window**:
  * <kbd>Enter</kbd> or <kbd>Esc</kbd>: Exit and close the camera window safely.
  * <kbd>M</kbd>: Toggle AI Gesture Recognition mode on/off.
* **Performance Dashboard**: While running, open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000) to see live FPS and latency metrics.

---

### Mode 2: Voice Assistant GUI ("Proton")
Launches the desktop GUI window and listens for voice commands.

```powershell
cd src
python Proton.py
```

* **Sample Voice Commands**:
  * `"Proton Launch Gesture Recognition"` &rarr; Starts the camera gesture mouse.
  * `"Proton Stop Gesture Recognition"` &rarr; Turns off the camera gesture mouse.
  * `"Proton search <topic>"` &rarr; Opens Google in Chrome and searches.
  * `"Proton open notepad"` &rarr; Launches Windows Notepad.
  * `"Proton open calculator"` &rarr; Launches Windows Calculator.
  * `"Proton what is the time"` &rarr; Speaks the current time.
  * `"Proton exit"` &rarr; Terminates the voice assistant.

---

### Mode 3: Glove-Controlled Virtual Mouse
For controlling the mouse using uniform-colored gloves (OpenCV color tracking):

```powershell
cd src
python Gesture_Controller_Gloved.py
```

---

## 4. Gesture Reference Card

| Gesture | Hand Pose | Mouse Action |
| :--- | :--- | :--- |
| **Neutral** | Open Palm (all 5 fingers up) | Halts cursor / stops current action |
| **Move Cursor** | Index + Middle fingertips touching/close | Moves the cursor across the screen |
| **Left Click** | Index finger pinch with thumb | Single left mouse click |
| **Right Click** | Middle finger folded / down | Single right mouse click |
| **Double Click** | Quick double pinch | Double left mouse click |
| **Scroll** | Pinch thumb & index, move hand up/down/left/right | Vertical or Horizontal scroll |
| **Drag & Drop** | Pinch and hold, move hand, release | Drag and drop files/windows |
| **Volume Control** | Minor hand pinch + move vertically | Increase / decrease system volume |
| **Brightness Control**| Minor hand pinch + move horizontally | Increase / decrease screen brightness |

---

## 5. Troubleshooting & FAQs

1. **"Webcam could not be opened" / Camera Error**:
   * Make sure another application (Zoom, Teams, Camera app) is not using your webcam.
   * If you have multiple cameras, specify your camera index:
     ```powershell
     $env:GESTURE_CAMERA_INDEX = "1"   # Try 0, 1, or 2
     python Gesture_Controller.py
     ```

2. **Microphone not detecting speech**:
   * Test your microphone input with the test script:
     ```powershell
     python mic_test.py
     ```
   * If using `Proton.py`, edit `MIC_DEVICE_INDEX` at line 58 of `src/Proton.py` to match your active microphone index.
