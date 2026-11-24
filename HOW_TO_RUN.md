# Quick Run & Setup Guide

This guide explains how to run the **Gesture-Controlled Virtual Mouse and Voice Assistant** on your current machine, as well as how to set it up on a new PC.

---

## ⚡ Option 1: 1-Click Launch (Easiest)

We have created two batch scripts at the project root for instant launch:

* **To run the Hand Gesture Virtual Mouse**:  
  Double-click `run_gesture_mouse.bat`
* **To run the Proton Voice Assistant GUI**:  
  Double-click `run_voice_assistant.bat`

*(These scripts automatically detect your virtual environment `.venv311` or `.venv` and run the project immediately).*

---

## 💻 Option 2: Running via Terminal (PowerShell)

### Step 1: Open PowerShell in the project directory
```powershell
cd "D:\Gesture Control\Gesture-Controlled-Virtual-Mouse\Gesture-Controlled-Virtual-Mouse"
```

### Step 2: Activate the Virtual Environment
On this computer (where Python 3.11 environment is pre-configured):
```powershell
.\.venv311\Scripts\Activate.ps1
```
*(On a fresh installation where `.venv` was created, use: `.\.venv\Scripts\Activate.ps1`)*

> **Note**: If PowerShell shows a script execution error, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```
> and then re-run the activation command.

### Step 3: Run the Application

#### Mode 1: Hand Gesture Virtual Mouse (Recommended)
```powershell
cd src
python Gesture_Controller.py
```
* **Controls**:
  * <kbd>Enter</kbd> or <kbd>Esc</kbd> in the camera window: Exit and close the webcam.
  * <kbd>M</kbd>: Toggle AI Gesture Recognition mode on/off.
* **Live Performance Dashboard**:  
  While running, open your web browser at [http://127.0.0.1:5000](http://127.0.0.1:5000) for real-time FPS and latency tracking.

#### Mode 2: Voice Assistant GUI ("Proton")
```powershell
cd src
python Proton.py
```
* **Sample Voice Commands**:
  * `"Proton Launch Gesture Recognition"` &rarr; Turns on the webcam gesture mouse.
  * `"Proton Stop Gesture Recognition"` &rarr; Stops the webcam gesture mouse.
  * `"Proton search <query>"` &rarr; Searches Google in Chrome.
  * `"Proton open notepad"` &rarr; Opens Windows Notepad.
  * `"Proton open calculator"` &rarr; Opens Calculator.
  * `"Proton what is the time"` &rarr; Speaks the current time.
  * `"Proton exit"` &rarr; Closes Proton.

#### Mode 3: Glove-Controlled Virtual Mouse
For controlling the mouse using uniform-colored gloves (OpenCV color tracking):
```powershell
cd src
python Gesture_Controller_Gloved.py
```

---

## 🖐️ Hand Gestures Reference

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

## 🛠️ Setting Up from Scratch on a New PC (For Friends)

If you clone this repository to a new computer, run these steps once:

1. **Install Python 3.11** ([python.org](https://www.python.org/downloads/)) — make sure to check *"Add python.exe to PATH"*.
2. **Clone the repository**:
   ```powershell
   git clone https://github.com/PKgamer1358/Virtual_Mouse.git
   cd Virtual_Mouse
   ```
3. **Create and activate a virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. **Install dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements-py311.txt
   pip install sounddevice vosk assemblyai
   ```
5. **(Optional) Offline Voice Model**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File setup_voice.ps1
   ```
6. **Run**:
   ```powershell
   cd src
   python Gesture_Controller.py
   ```

---

## ❓ Troubleshooting

1. **Camera Not Opening**:
   * Ensure no other program (Zoom, Teams, Camera app) is currently using the camera.
   * If you have multiple cameras, specify your camera index:
     ```powershell
     $env:GESTURE_CAMERA_INDEX = "1"   # Try 0, 1, or 2
     python Gesture_Controller.py
     ```
2. **Microphone Issues**:
   * Run `python mic_test.py` from the project root to test microphone input.
