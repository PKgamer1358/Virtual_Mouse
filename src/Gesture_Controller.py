# Imports
import time
import os
import sys
import subprocess

from performance_monitor import (
    start_dashboard,
    record_frame,
    record_click,
)
from custom_gestures import extract_features, load_model
import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbcontrol
from voice_assistant import VoiceAssistant

pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Gesture Encodings 
class Gest(IntEnum):
    """
    Enum for mapping all hand gesture to binary number.
    """
    # Binary Encoded

    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16    
    PALM = 31
    
    # Extra Mappings
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36

# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

# Convert Mediapipe Landmarks to recognizable Gestures
class HandRecog:
    """
    Convert Mediapipe Landmarks to recognizable Gestures.
    """
    
    def __init__(self, hand_label):
        """
        Constructs all the necessary attributes for the HandRecog object.

        Parameters
        ----------
            finger : int
                Represent gesture corresponding to Enum 'Gest',
                stores computed gesture for current frame.
            ori_gesture : int
                Represent gesture corresponding to Enum 'Gest',
                stores gesture being used.
            prev_gesture : int
                Represent gesture corresponding to Enum 'Gest',
                stores gesture computed for previous frame.
            frame_count : int
                total no. of frames since 'ori_gesture' is updated.
            hand_result : Object
                Landmarks obtained from mediapipe.
            hand_label : int
                Represents multi-handedness corresponding to Enum 'HLabel'.
        """

        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
    
    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_signed_dist(self, point):
        """
        returns signed euclidean distance between 'point'.

        Parameters
        ----------
        point : list contaning two elements of type list/tuple which represents 
            landmark point.
        
        Returns
        -------
        float
        """
        assert self.hand_result is not None
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist*sign
    
    def get_dist(self, point):
        """
        returns euclidean distance between 'point'.

        Parameters
        ----------
        point : list contaning two elements of type list/tuple which represents 
            landmark point.
        
        Returns
        -------
        float
        """
        assert self.hand_result is not None
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist
    
    def get_dz(self,point):
        """
        returns absolute difference on z-axis between 'point'.

        Parameters
        ----------
        point : list contaning two elements of type list/tuple which represents 
            landmark point.
        
        Returns
        -------
        float
        """
        assert self.hand_result is not None
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)
    
    # Function to find Gesture Encoding using current finger_state.
    # Finger_state: 1 if finger is open, else 0
    def set_finger_state(self):
        """
        set 'finger' by computing ratio of distance between finger tip 
        , middle knuckle, base knuckle.

        Returns
        -------
        None
        """
        if self.hand_result is None:
            return

        points = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]
        self.finger = 0
        self.finger = self.finger | 0 #thumb
        for idx,point in enumerate(points):
            
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            
            try:
                ratio = round(dist/dist2,1)
            except:
                ratio = round(dist/0.01,1)

            self.finger = self.finger << 1
            if ratio > 0.5 :
                self.finger = self.finger | 1
    

    # Handling Fluctations due to noise
    
    
    def get_gesture(self):
        """
        Returns the gesture corresponding to the current finger state.
        Handles fluctuations and stabilizes recognition across frames.
        """
        if self.hand_result is None:
            return Gest.PALM

        current_gesture = Gest.PALM

        if self.finger in [Gest.LAST3, Gest.LAST4] and self.get_dist([8, 4]) < 0.05:
            if self.hand_label == HLabel.MINOR:
                current_gesture = Gest.PINCH_MINOR
            else:
                current_gesture = Gest.PINCH_MAJOR

        elif Gest.FIRST2 == self.finger:
            point = [[8, 12], [5, 9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])

            if dist2 == 0:
                ratio = 0
            else:
                ratio = dist1 / dist2

            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            else:
                if self.get_dz([8, 12]) < 0.1:
                    current_gesture = Gest.TWO_FINGER_CLOSED
                else:
                    current_gesture = Gest.MID

        else:
            try:
                current_gesture = Gest(self.finger)
            except ValueError:
                # Unknown finger pattern: neutral gesture.
                current_gesture = Gest.PALM

        # Gesture stabilization across frames.
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        if self.frame_count > 2:
            self.ori_gesture = current_gesture

        return self.ori_gesture



# Executes commands according to detected gestures
class Controller:
    """
    Executes commands according to detected gestures.

    Attributes
    ----------
    tx_old : int
        previous mouse location x coordinate
    ty_old : int
        previous mouse location y coordinate
    flag : bool
        true if V gesture is detected
    grabflag : bool
        true if FIST gesture is detected
    pinchmajorflag : bool
        true if PINCH gesture is detected through MAJOR hand,
        on x-axis 'Controller.changesystembrightness', 
        on y-axis 'Controller.changesystemvolume'.
    pinchminorflag : bool
        true if PINCH gesture is detected through MINOR hand,
        on x-axis 'Controller.scrollHorizontal', 
        on y-axis 'Controller.scrollVertical'.
    pinchstartxcoord : int
        x coordinate of hand landmark when pinch gesture is started.
    pinchstartycoord : int
        y coordinate of hand landmark when pinch gesture is started.
    pinchdirectionflag : bool
        true if pinch gesture movment is along x-axis,
        otherwise false
    prevpinchlv : int
        stores quantized magnitued of prev pinch gesture displacment, from 
        starting position
    pinchlv : int
        stores quantized magnitued of pinch gesture displacment, from 
        starting position
    framecount : int
        stores no. of frames since 'pinchlv' is updated.
    prev_hand : tuple
        stores (x, y) coordinates of hand in previous frame.
    pinch_threshold : float
        step size for quantization of 'pinchlv'.
    """

    tx_old = 0
    ty_old = 0
    trial = True
    flag = False
    grabflag = False
    pinchmajorflag = False
    pinchminorflag = False
    pinchstartxcoord = None
    pinchstartycoord = None
    pinchdirectionflag = None
    prevpinchlv = 0
    pinchlv = 0
    framecount = 0
    prev_hand = None

    # Intelligent cursor stabilization settings
    smoothing_alpha = 0.35
    movement_deadzone = 1.0

    # AI-mode two-finger (FIRST2) scrolling settings.
    # MediaPipe landmark 9 tracks the hand's vertical movement.
    scroll_prev_y = None
    scroll_deadzone = 0.0025
    scroll_sensitivity = 1600
    max_scroll_per_frame = 60

    pinch_threshold = 0.3

    @staticmethod
    def scroll_with_two_fingers(hand_result):
        """Scroll vertically while the FIRST2 gesture is recognized in AI mode."""
        current_y = float(hand_result.landmark[9].y)

        if Controller.scroll_prev_y is None:
            Controller.scroll_prev_y = current_y
            return

        delta_y = current_y - Controller.scroll_prev_y
        Controller.scroll_prev_y = current_y

        if abs(delta_y) < Controller.scroll_deadzone:
            return

        # Moving the hand upward scrolls up; downward scrolls down.
        amount = int(-delta_y * Controller.scroll_sensitivity)
        amount = max(
            -Controller.max_scroll_per_frame,
            min(amount, Controller.max_scroll_per_frame)
        )

        if amount:
            pyautogui.scroll(amount)

    @staticmethod
    def reset_palm_scroll():
        """Reset scroll tracking when FIRST2 is no longer active."""
        Controller.scroll_prev_y = None

    @staticmethod
    def reset_scroll_tracking():
        """Compatibility method used by handle_controls to reset two-finger scrolling."""
        Controller.reset_palm_scroll()
    
    @staticmethod
    def getpinchylv(hand_result):
        """returns distance beween starting pinch y coord and current hand position y coord."""
        dist = round((Controller.pinchstartycoord - hand_result.landmark[8].y)*10,1)
        return dist

    @staticmethod
    def getpinchxlv(hand_result):
        """returns distance beween starting pinch x coord and current hand position x coord."""
        dist = round((hand_result.landmark[8].x - Controller.pinchstartxcoord)*10,1)
        return dist
    
    @staticmethod
    def changesystembrightness():
        """sets system brightness based on 'Controller.pinchlv'."""
        _b = sbcontrol.get_brightness(display=0)
        _b_val: int = _b[0] if isinstance(_b, list) else _b  # type: ignore[index]
        currentBrightnessLv = _b_val / 100.0
        currentBrightnessLv += Controller.pinchlv/50.0
        if currentBrightnessLv > 1.0:
            currentBrightnessLv = 1.0
        elif currentBrightnessLv < 0.0:
            currentBrightnessLv = 0.0
        _start = sbcontrol.get_brightness(display=0)
        _start_val: int = _start[0] if isinstance(_start, list) else _start  # type: ignore[index]
        sbcontrol.fade_brightness(int(100 * currentBrightnessLv), start=_start_val)
    
    @staticmethod
    def changesystemvolume():
        """sets system volume based on 'Controller.pinchlv'."""
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))  # type: ignore[arg-type]
        currentVolumeLv = volume.GetMasterVolumeLevelScalar()
        currentVolumeLv += Controller.pinchlv/50.0
        if currentVolumeLv > 1.0:
            currentVolumeLv = 1.0
        elif currentVolumeLv < 0.0:
            currentVolumeLv = 0.0
        volume.SetMasterVolumeLevelScalar(currentVolumeLv, None)
    
    @staticmethod
    def scrollVertical():
        """scrolls on screen vertically."""
        pyautogui.scroll(120 if Controller.pinchlv>0.0 else -120)
        
    
    @staticmethod
    def scrollHorizontal():
        """scrolls on screen horizontally."""
        pyautogui.keyDown('shift')
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-120 if Controller.pinchlv>0.0 else 120)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')

    # Locate Hand to get Cursor Position
    
    # Locate hand and calculate stabilized cursor position
    @staticmethod
    def get_position(hand_result):
        """Calculate cursor position using EMA stabilization."""

        if hand_result is None:
            return pyautogui.position()

        point = 9
        raw_x = hand_result.landmark[point].x
        raw_y = hand_result.landmark[point].y

        screen_width, screen_height = pyautogui.size()

        raw_x *= screen_width
        raw_y *= screen_height

        # Initialize the filter without jumping the cursor.
        if Controller.prev_hand is None:
            Controller.prev_hand = [raw_x, raw_y]
            return pyautogui.position()

        # Exponential Moving Average.
        alpha = Controller.smoothing_alpha

        smooth_x = (
            alpha * raw_x
            + (1.0 - alpha) * Controller.prev_hand[0]
        )
        smooth_y = (
            alpha * raw_y
            + (1.0 - alpha) * Controller.prev_hand[1]
        )

        delta_x = smooth_x - Controller.prev_hand[0]
        delta_y = smooth_y - Controller.prev_hand[1]

        Controller.prev_hand = [smooth_x, smooth_y]

        # Ignore tiny movements.
        distance_squared = delta_x ** 2 + delta_y ** 2
        deadzone = Controller.movement_deadzone

        if distance_squared < deadzone ** 2:
            return pyautogui.position()

        distance = math.sqrt(distance_squared)

        # Preserve the original adaptive sensitivity.
        if distance_squared <= 25:
            ratio = 1.0
        elif distance_squared <= 900:
            ratio = 0.07 * distance
        else:
            ratio = 2.1

        cursor_x, cursor_y = pyautogui.position()

        target_x = int(cursor_x + delta_x * ratio)
        target_y = int(cursor_y + delta_y * ratio)

        # Keep cursor coordinates within the screen.
        target_x = max(0, min(target_x, screen_width - 1))
        target_y = max(0, min(target_y, screen_height - 1))

        return target_x, target_y


    @staticmethod
    def pinch_control_init(hand_result):
        """Initializes attributes for pinch gesture."""
        Controller.pinchstartxcoord = hand_result.landmark[8].x
        Controller.pinchstartycoord = hand_result.landmark[8].y
        Controller.pinchlv = 0
        Controller.prevpinchlv = 0
        Controller.framecount = 0

    # Hold final position for 5 frames to change status
    @staticmethod
    def pinch_control(hand_result, controlHorizontal, controlVertical):
        """
        calls 'controlHorizontal' or 'controlVertical' based on pinch flags, 
        'framecount' and sets 'pinchlv'.

        Parameters
        ----------
        hand_result : Object
            Landmarks obtained from mediapipe.
        controlHorizontal : callback function assosiated with horizontal
            pinch gesture.
        controlVertical : callback function assosiated with vertical
            pinch gesture. 
        
        Returns
        -------
        None
        """
        if Controller.framecount == 5:
            Controller.framecount = 0
            Controller.pinchlv = Controller.prevpinchlv

            if Controller.pinchdirectionflag is True:
                controlHorizontal() #x

            elif Controller.pinchdirectionflag is False:
                controlVertical() #y

        lvx =  Controller.getpinchxlv(hand_result)
        lvy =  Controller.getpinchylv(hand_result)
            
        if abs(lvy) > abs(lvx) and abs(lvy) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = False
            if abs(Controller.prevpinchlv - lvy) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvy
                Controller.framecount = 0

        elif abs(lvx) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = True
            if abs(Controller.prevpinchlv - lvx) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvx
                Controller.framecount = 0

    @staticmethod
    def handle_controls(gesture, hand_result):
        """Execute mouse and system controls for the recognized gesture."""
        if hand_result is None:
            return

        x, y = None, None
        if gesture != Gest.FIRST2:
            x, y = Controller.get_position(hand_result)

        # Release a drag when the fist gesture ends.
        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button="left")

        # Reset scrolling when the two-finger gesture ends.
        if gesture != Gest.FIRST2:
            Controller.reset_scroll_tracking()

        # Reset pinch states when the corresponding pinch ends.
        if gesture != Gest.PINCH_MAJOR:
            Controller.pinchmajorflag = False
        if gesture != Gest.PINCH_MINOR:
            Controller.pinchminorflag = False

        # Two extended fingers: continuous vertical scrolling in AI mode.
        if gesture == Gest.FIRST2:
            Controller.scroll_with_two_fingers(hand_result)

        # Move the cursor with the V gesture.
        elif gesture == Gest.V_GEST:
            Controller.flag = True
            if x is not None and y is not None:
                pyautogui.moveTo(x, y, duration=0.03)

        # Fist: hold left mouse button and drag.
        elif gesture == Gest.FIST:
            if not Controller.grabflag:
                Controller.grabflag = True
                pyautogui.mouseDown(button="left")
            if x is not None and y is not None:
                pyautogui.moveTo(x, y, duration=0.03)

        elif gesture == Gest.MID and Controller.flag:
            pyautogui.click(button="left")
            record_click("left")
            Controller.flag = False

        elif gesture == Gest.INDEX and Controller.flag:
            pyautogui.click(button="right")
            record_click("right")
            Controller.flag = False

        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            pyautogui.doubleClick()
            record_click("double")
            Controller.flag = False

        elif gesture == Gest.PINCH_MINOR:
            if not Controller.pinchminorflag:
                Controller.pinch_control_init(hand_result)
                Controller.pinchminorflag = True
            Controller.pinch_control(
                hand_result,
                Controller.scrollHorizontal,
                Controller.scrollVertical
            )

        elif gesture == Gest.PINCH_MAJOR:
            if not Controller.pinchmajorflag:
                Controller.pinch_control_init(hand_result)
                Controller.pinchmajorflag = True
            Controller.pinch_control(
                hand_result,
                Controller.changesystembrightness,
                Controller.changesystemvolume
            )

'''
----------------------------------------  Main Class  ----------------------------------------
    Entry point of Gesture Controller
'''


class GestureController:
    """
    Handles camera, obtain landmarks from mediapipe, entry point
    for whole program.

    Attributes
    ----------
    gc_mode : int
        indicates weather gesture controller is running or not,
        1 if running, otherwise 0.
    cap : Object
        object obtained from cv2, for capturing video frame.
    CAM_HEIGHT : int
        highet in pixels of obtained frame from camera.
    CAM_WIDTH : int
        width in pixels of obtained frame from camera.
    hr_major : Object of 'HandRecog'
        object representing major hand.
    hr_minor : Object of 'HandRecog'
        object representing minor hand.
    dom_hand : bool
        True if right hand is domaniant hand, otherwise False.
        default True.
    """
    gc_mode = 0
    cap: cv2.VideoCapture | None = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    hr_major = None # Right Hand by default
    hr_minor = None # Left hand by default
    dom_hand = True

    
    def __init__(self):
        """Initialize the webcam using the Windows DirectShow backend."""
        GestureController.gc_mode = 1

        camera_index = int(os.environ.get("GESTURE_CAMERA_INDEX", "0"))

        if sys.platform == "win32":
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            cap.release()
            GestureController.gc_mode = 0
            raise RuntimeError(
                f"Could not open webcam at index {camera_index}. "
                "Check camera permissions or try another camera index."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        success, frame = cap.read()

        if not success or frame is None:
            cap.release()
            GestureController.gc_mode = 0
            raise RuntimeError("Webcam opened, but could not capture a frame.")

        GestureController.cap = cap
        GestureController.CAM_HEIGHT = frame.shape[0]
        GestureController.CAM_WIDTH = frame.shape[1]

        print("[CAMERA] Webcam initialized successfully.")
        print(
            f"[CAMERA] Resolution: "
            f"{GestureController.CAM_WIDTH}x{GestureController.CAM_HEIGHT}"
        )

    
    @staticmethod
    def classify_hands(results):
        """
        sets 'hr_major', 'hr_minor' based on classification(left, right) of 
        hand obtained from mediapipe, uses 'dom_hand' to decide major and
        minor hand.
        """
        left , right = None,None
        try:
            handedness_dict = MessageToDict(results.multi_handedness[0])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[0]
            else :
                left = results.multi_hand_landmarks[0]
        except:
            pass

        try:
            handedness_dict = MessageToDict(results.multi_handedness[1])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[1]
            else :
                left = results.multi_hand_landmarks[1]
        except:
            pass
        
        if GestureController.dom_hand is True:
            GestureController.hr_major = right
            GestureController.hr_minor = left
        else :
            GestureController.hr_major = left
            GestureController.hr_minor = right

        
    def start(self):
        """
        Captures webcam frames, recognizes hand gestures, and optionally
        uses the personalized AI gesture model.
        """
        start_dashboard()
        print("Performance dashboard: http://127.0.0.1:5000")
        voice_assistant = VoiceAssistant()
        try:
            voice_assistant.start()
        except Exception as exc:
            print("[VOICE] Voice assistant not started:", exc)
            voice_assistant = None

        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)

        ai_model = load_model()
        ai_mode = False

        keyboard_process = None
        keyboard_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "virtual_keyboard.py",
        )

        if ai_model is None:
            print("AI model not found. Using original gesture recognition.")
            print("Collect gesture data and train the model to enable AI mode.")
        else:
            print("AI gesture model loaded.")
            print("Press M in the Gesture Controller window to toggle AI mode.")

        try:
            with mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as hands:

                assert GestureController.cap is not None
                while (
                    GestureController.cap.isOpened()
                    and GestureController.gc_mode
                ):
                    frame_started = time.perf_counter()
                    success, image = GestureController.cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        continue

                    image = cv2.cvtColor(
                        cv2.flip(image, 1),
                        cv2.COLOR_BGR2RGB
                    )
                    image.flags.writeable = False
                    recognition_started = time.perf_counter()
                    results = hands.process(image)

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    display_text = "AI MODE: OFF"

                    if results.multi_hand_landmarks:  # type: ignore[union-attr]
                        GestureController.classify_hands(results)

                        handmajor.update_hand_result(
                            GestureController.hr_major
                        )
                        handminor.update_hand_result(
                            GestureController.hr_minor
                        )

                        if ai_mode and ai_model is not None:
                            landmarks = GestureController.hr_major

                            if landmarks is not None:
                                features = extract_features(landmarks)
                                probabilities = ai_model.predict_proba(
                                    [features]
                                )[0]
                                best_index = int(probabilities.argmax())
                                confidence = float(probabilities[best_index])
                                label = str(ai_model.classes_[best_index])

                                gesture_map = {
                                    "FIRST2": Gest.FIRST2,
                                    "FIST": Gest.FIST,
                                    "INDEX": Gest.INDEX,
                                    "PALM": Gest.PALM,
                                    "V_GEST": Gest.V_GEST,
                                }

                                if confidence >= 0.75 and label in gesture_map:
                                    predicted_gesture = gesture_map[label]
                                    Controller.handle_controls(
                                        predicted_gesture,
                                        landmarks
                                    )
                                    display_text = f"AI: {label} ({confidence:.0%})"
                                else:
                                    # Release an active drag if recognition becomes uncertain.
                                    if Controller.grabflag:
                                        pyautogui.mouseUp(button="left")
                                        Controller.grabflag = False
                                    Controller.reset_scroll_tracking()
                                    display_text = f"AI: {label} - LOW CONFIDENCE ({confidence:.0%})"
                            else:
                                display_text = "AI: Show your dominant hand"

                        else:
                            handmajor.set_finger_state()
                            handminor.set_finger_state()

                            gest_name = handminor.get_gesture()

                            if gest_name == Gest.PINCH_MINOR:
                                Controller.handle_controls(
                                    gest_name,
                                    handminor.hand_result
                                )
                            else:
                                gest_name = handmajor.get_gesture()
                                Controller.handle_controls(
                                    gest_name,
                                    handmajor.hand_result
                                )

                        for hand_landmarks in results.multi_hand_landmarks:  # type: ignore[union-attr]
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                list(mp_hands.HAND_CONNECTIONS)  # type: ignore[arg-type]
                            )

                    else:
                        Controller.prev_hand = None
                        Controller.reset_scroll_tracking()
                        if Controller.grabflag:
                            pyautogui.mouseUp(button="left")
                            Controller.grabflag = False

                    record_frame(
                        frame_seconds=time.perf_counter() - frame_started,
                        latency_ms=(time.perf_counter() - recognition_started) * 1000,
                        gesture=display_text,
                    )
                    cv2.putText(
                        image,
                        display_text,
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0) if ai_mode else (255, 255, 0),
                        2
                    )

                    cv2.imshow("Gesture Controller", image)

                    key = cv2.waitKey(5) & 0xFF

                    # K: toggle the floating virtual keyboard open / closed.
                    if key in (ord("k"), ord("K")):
                        if (
                            keyboard_process is None
                            or keyboard_process.poll() is not None
                        ):
                            keyboard_process = subprocess.Popen(
                                [sys.executable, keyboard_path],
                                cwd=os.path.dirname(os.path.abspath(__file__)),
                            )
                            print("Virtual keyboard opened.")
                        else:
                            keyboard_process.terminate()
                            keyboard_process.wait(timeout=3)
                            keyboard_process = None
                            print("Virtual keyboard closed.")

                    if key == ord("m"):
                        if ai_model is None:
                            print(
                                "AI model unavailable. "
                                "Train the model first."
                            )
                        else:
                            ai_mode = not ai_mode
                            print(
                                "AI mode:",
                                "ON" if ai_mode else "OFF"
                            )

                    if key == 13 or key == 27:
                        break

        finally:
            GestureController.gc_mode = 0

            # Safety: release the left mouse button if shutdown occurs mid-drag.
            if Controller.grabflag:
                pyautogui.mouseUp(button="left")
                Controller.grabflag = False

            if GestureController.cap is not None:
                GestureController.cap.release()

            if (
                keyboard_process is not None
                and keyboard_process.poll() is None
            ):
                keyboard_process.terminate()
            if voice_assistant is not None:
                voice_assistant.stop()
                
            cv2.destroyAllWindows()



if __name__ == "__main__":
    gc1 = GestureController()
    gc1.start()
