import numpy as np
import cv2
import cv2.aruco as aruco
import os
import glob
import math
import pyautogui
import time

class Marker:
    """ArUco marker detector compatible with old and new OpenCV contrib APIs."""

    def __init__(self, dict_type=None, thresh_constant=1):
        if not hasattr(aruco, "getPredefinedDictionary") and not hasattr(aruco, "Dictionary_get"):
            raise RuntimeError(
                "cv2.aruco is unavailable. Install opencv-contrib-python in the active venv."
            )

        if dict_type is None:
            dict_type = getattr(aruco, "DICT_4X4_50", 0)

        if hasattr(aruco, "getPredefinedDictionary"):
            self.aruco_dict = aruco.getPredefinedDictionary(dict_type)
        else:
            self.aruco_dict = aruco.Dictionary_get(dict_type)

        if hasattr(aruco, "DetectorParameters"):
            self.parameters = aruco.DetectorParameters()
        else:
            self.parameters = aruco.DetectorParameters_create()

        self.parameters.adaptiveThreshConstant = thresh_constant
        self.detector = None
        if hasattr(aruco, "ArucoDetector"):
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.corners = None
        self.ids = None
        self.marker_x2y = 1.0
        self.mtx, self.dist = self.calibrate()

    @staticmethod
    def calibrate():
        """Calibrate from checkerboard images; use a safe fallback if none exist."""
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        image_size = None

        base_path = os.path.dirname(os.path.abspath(__file__))
        pattern = os.path.join(base_path, "calib_images", "checkerboard", "*.jpg")

        for fname in glob.glob(pattern):
            img = cv2.imread(fname)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, (7, 6), None)
            if found and corners is not None:
                objpoints.append(objp.copy())
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                imgpoints.append(refined)

        if objpoints and imgpoints and image_size is not None:
            _ret, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, image_size, None, None
            )
            return camera_matrix, dist_coeffs

        # Fallback allows marker detection without calibration images.
        # Pose estimation will be approximate until real calibration is supplied.
        camera_matrix = np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        print("WARNING: No valid checkerboard calibration images found; using fallback camera parameters.")
        return camera_matrix, dist_coeffs

    def detect(self, frame):
        if frame is None:
            self.corners, self.ids = None, None
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, _rejected = self.detector.detectMarkers(gray_frame)
        else:
            corners, ids, _rejected = aruco.detectMarkers(
                gray_frame, self.aruco_dict, parameters=self.parameters
            )

        if ids is not None and len(ids) > 0:
            self.corners, self.ids = corners, ids
            # Pose estimation is optional for cursor tracking; guard API differences.
            if hasattr(aruco, "estimatePoseSingleMarkers"):
                try:
                    aruco.estimatePoseSingleMarkers(
                        self.corners, 0.05, self.mtx, self.dist
                    )
                except cv2.error:
                    pass
        else:
            self.corners, self.ids = None, None

    def is_detected(self):
        return self.corners is not None and len(self.corners) > 0

    def draw_marker(self, frame):
        if self.is_detected():
            aruco.drawDetectedMarkers(frame, self.corners)


def ecu_dis(p1, p2):
    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist

def find_HSV(samples):
    try:
        color = np.uint8([ samples ])
    except:
        color = np.uint8([ [[105,105,50]] ])
    hsv_color = cv2.cvtColor(color,cv2.COLOR_RGB2HSV)
    #print( hsv_color )
    return hsv_color

def draw_box(frame, points, color=(0,255,127)):
    if points:
        frame = cv2.line(frame, points[0], points[1], color, thickness=2, lineType=8) #top
        frame = cv2.line(frame, points[1], points[2], color, thickness=2, lineType=8) #right
        frame = cv2.line(frame, points[2], points[3], color, thickness=2, lineType=8) #bottom
        frame = cv2.line(frame, points[3], points[0], color, thickness=2, lineType=8) #left

def in_cam(val, type_):
    if type_ == 'x':
        if val<0:
            return 0
        if val>GestureController.cam_width:
            return GestureController.cam_width
    elif type_ == 'y':
        if val<0:
            return 0
        if val>GestureController.cam_height:
            return GestureController.cam_height
    return val

    
class ROI:
    def __init__(self, roi_alpha1=1.5, roi_alpha2=1.5, roi_beta=2.5, hsv_alpha = 0.3, hsv_beta = 0.5, hsv_lift_up = 0.3):
        self.roi_alpha1 = roi_alpha1
        self.roi_alpha2 = roi_alpha2
        self.roi_beta = roi_beta
        self.roi_corners = None
        
        self.hsv_alpha = hsv_alpha
        self.hsv_beta = hsv_beta
        self.hsv_lift_up = hsv_lift_up
        self.hsv_corners = None
        
        self.marker_top = None
        self.glove_hsv = None
        
    def findROI(self, frame, marker):
        rec_coor = marker.corners[0][0]
        c1 = (int(rec_coor[0][0]),int(rec_coor[0][1]))
        c2 = (int(rec_coor[1][0]),int(rec_coor[1][1]))
        c3 = (int(rec_coor[2][0]),int(rec_coor[2][1]))
        c4 = (int(rec_coor[3][0]),int(rec_coor[3][1]))
        
        try:
            marker.marker_x2y = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) / np.sqrt((c3[0]-c2[0])**2 + (c3[1]-c2[1])**2)
        except:
            marker.marker_x2y = 999.0
        
        #mid-point of top line of Marker
        cx = (c1[0] + c2[0])/2
        cy = (c1[1] + c2[1])/2
        
        self.marker_top = [cx, cy]
        
        l = np.absolute(ecu_dis(c1,c4))
        
        try:
            slope_12 = (c1[1]-c2[1])/(c1[0]-c2[0])
        except:
            slope_12 = (c1[1]-c2[1])*999.0 + 0.1
        
        try:
            slope_14 = -1 / slope_12
        except:
            slope_14 = -999.0
        
        if slope_14 < 0:
            sign = 1
        else:
            sign = -1
        
        bot_rx = int(cx + self.roi_alpha2 * l * np.sqrt(1/(1+slope_12**2)))
        bot_ry = int(cy + self.roi_alpha2 * slope_12 * l * np.sqrt(1/(1+slope_12**2)))
        
        bot_lx = int(cx - self.roi_alpha1 * l * np.sqrt(1/(1+slope_12**2)))
        bot_ly = int(cy - self.roi_alpha1 * slope_12 * l * np.sqrt(1/(1+slope_12**2)))
        
        top_lx = int(bot_lx + sign * self.roi_beta * l * np.sqrt(1/(1+slope_14**2)))
        top_ly = int(bot_ly + sign * self.roi_beta * slope_14 * l * np.sqrt(1/(1+slope_14**2)))
        
        top_rx = int(bot_rx + sign * self.roi_beta * l * np.sqrt(1/(1+slope_14**2)))
        top_ry = int(bot_ry + sign * self.roi_beta * slope_14 * l * np.sqrt(1/(1+slope_14**2)))
        
        bot_lx = in_cam(bot_lx, 'x')
        bot_ly = in_cam(bot_ly, 'y')
        
        bot_rx = in_cam(bot_rx, 'x')
        bot_ry = in_cam(bot_ry, 'y')
        
        top_lx = in_cam(top_lx, 'x')
        top_ly = in_cam(top_ly, 'y')
        
        top_rx = in_cam(top_rx, 'x')
        top_ry = in_cam(top_ry, 'y')
        
        self.roi_corners = [(bot_lx,bot_ly), (bot_rx,bot_ry), (top_rx,top_ry), (top_lx,top_ly)]
        
        
    def find_glove_hsv(self, frame, marker):
        rec_coor = marker.corners[0][0]
        c1 = (int(rec_coor[0][0]),int(rec_coor[0][1]))
        c2 = (int(rec_coor[1][0]),int(rec_coor[1][1]))
        c3 = (int(rec_coor[2][0]),int(rec_coor[2][1]))
        c4 = (int(rec_coor[3][0]),int(rec_coor[3][1]))
        
        l = np.absolute(ecu_dis(c1,c4))
        
        try:
            slope_12 = (c1[1]-c2[1])/(c1[0]-c2[0])
        except:
            slope_12 = (c1[1]-c2[1])*999.0 + 0.1
        try:
            slope_14 = -1 / slope_12
        except:
            slope_14 = -999.0
        
        if slope_14 < 0:
            sign = 1
        else:
            sign = -1
               
        bot_rx = int(self.marker_top[0] + self.hsv_alpha * l * np.sqrt(1/(1+slope_12**2)))
        bot_ry = int(self.marker_top[1] - self.hsv_lift_up*l + self.hsv_alpha * slope_12 * l * np.sqrt(1/(1+slope_12**2)))
        
        bot_lx = int(self.marker_top[0] - self.hsv_alpha * l * np.sqrt(1/(1+slope_12**2)))
        bot_ly = int(self.marker_top[1] - self.hsv_lift_up*l - self.hsv_alpha * slope_12 * l * np.sqrt(1/(1+slope_12**2)))
        
        top_lx = int(bot_lx + sign * self.hsv_beta * l * np.sqrt(1/(1+slope_14**2)))
        top_ly = int(bot_ly + sign * self.hsv_beta * slope_14 * l * np.sqrt(1/(1+slope_14**2)))
        
        top_rx = int(bot_rx + sign * self.hsv_beta * l * np.sqrt(1/(1+slope_14**2)))
        top_ry = int(bot_ry + sign * self.hsv_beta * slope_14 * l * np.sqrt(1/(1+slope_14**2)))
        
        height, width = frame.shape[:2]
        x1, x2 = sorted((max(0, top_lx), min(width, bot_rx)))
        y1, y2 = sorted((max(0, top_ry), min(height, bot_ry)))
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            region = frame[max(0, int(self.marker_top[1])-5):int(self.marker_top[1])+5,
                           max(0, int(self.marker_top[0])-5):int(self.marker_top[0])+5]
        if region.size == 0:
            self.hsv_glove = find_HSV([[[105, 105, 50]]])
        else:
            b, g, r = np.mean(region, axis=(0, 1))
            self.hsv_glove = find_HSV([[r, g, b]])
        self.hsv_corners =  [(bot_lx,bot_ly), (bot_rx,bot_ry), (top_rx,top_ry), (top_lx,top_ly)]
        
    
    def cropROI(self, frame):
        pts = np.array(self.roi_corners)
        
        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        croped = frame[y:y+h, x:x+w].copy()
        
        ## (2) make mask
        pts = pts - pts.min(axis=0)
        
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        
        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        
        ## (4) add the white background
        bg = np.ones_like(croped, np.uint8)*255
        cv2.bitwise_not(bg,bg, mask=mask)
        
        kernelOpen = np.ones((3,3),np.uint8)
        kernelClose = np.ones((5,5),np.uint8)
        
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        
        lower_range = np.array([self.hsv_glove[0][0][0]//1-5,50,50])
        upper_range = np.array([self.hsv_glove[0][0][0]//1+5,255,255])
        
        mask = cv2.inRange(hsv, lower_range, upper_range)
        #mask = cv2.dilate(mask,kernelOpen,iterations = 1)
        Opening =cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
        Closing =cv2.morphologyEx(Opening,cv2.MORPH_CLOSE,kernelClose)
        FinalMask = Closing
        
        return FinalMask


class Glove:
    
    def __init__(self):
        self.fingers = 0
        self.arearatio = 0
        self.gesture = 0
    
    def find_fingers(self, FinalMask):
        self.fingers = 0
        self.arearatio = 0.0
        defects = None
        approx = None
        conts, _hierarchy = cv2.findContours(
            FinalMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not conts:
            return

        try:
            cnt = max(conts, key = lambda x: cv2.contourArea(x))
            #approx the contour a little
            epsilon = 0.0005*cv2.arcLength(cnt,True)
            approx= cv2.approxPolyDP(cnt,epsilon,True)
            #make convex hull around hand
            hull = cv2.convexHull(cnt)
            #define area of hull and area of hand
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
            #find the percentage of area not covered by hand in convex hull
            if areacnt <= 0:
                return
            self.arearatio = ((areahull - areacnt) / areacnt) * 100
            #find the defects in convex hull with respect to hand
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)
        except:
            print("No Contours found in FinalMask")
        
        # l = no. of defects
        l=0
        try:
            #code for finding no. of defects due to fingers
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                
                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                s = (a+b+c)/2
                ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                
                #distance between point and convex hull
                if a <= 0 or b <= 0 or c <= 0:
                    continue
                d = (2 * ar) / a

                # Clamp floating-point drift before acos.
                cosine = (b**2 + c**2 - a**2) / (2 * b * c)
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                      
                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                if angle <= 90 and d>30:
                    l += 1
                    #cv2.circle(frame, far, 3, [255,255,255], -1)
                
                #draw lines around hand
                cv2.line(FinalMask,start, end, [255,255,255], 2)
                
            l+=1
        except:
            l = 0
            print("No Defects found in mask")
        
        self.fingers = l
        
    def find_gesture(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.gesture = 0
        if self.fingers==1:
            #cv2.putText(frame, str(int(arearatio)), (10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            if self.arearatio<15:
                cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                self.gesture = 0
            elif self.arearatio<25:
                cv2.putText(frame,'2 fingers',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                self.gesture = 2
            else:
                cv2.putText(frame,'1 finger',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                self.gesture = 1
                    
        elif self.fingers==2:
            cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            self.gesture = 3
        '''
        elif self.fingers==3:
            #cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif self.fingers==4:
            #cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif self.fingers==5:
            #cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        else :
           # cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        '''

class Tracker:
    def __init__(self):
        self.tracker_started = False
        self.tracker = None
        self.start_time = 0.0
        self.now_time = 0.0
        self.tracker_bbox = None
        
    def corners_to_tracker(self, corners):
        csrt_minX = int( min( [corners[0][0][0][0], corners[0][0][1][0], corners[0][0][2][0], corners[0][0][3][0]] ))
        csrt_maxX = int( max( [corners[0][0][0][0], corners[0][0][1][0], corners[0][0][2][0], corners[0][0][3][0]] ))
        csrt_minY = int( min( [corners[0][0][0][1], corners[0][0][1][1], corners[0][0][2][1], corners[0][0][3][1]] ))
        csrt_maxY = int( max( [corners[0][0][0][1], corners[0][0][1][1], corners[0][0][2][1], corners[0][0][3][1]] ))
        self.tracker_bbox = [csrt_minX, csrt_minY, csrt_maxX-csrt_minX, csrt_maxY-csrt_minY]
        
    def tracker_to_corner(self, final_bbox):
        if self.tracker_bbox == None:
            return None
        x, y, width, height = self.tracker_bbox
        final_bbox = [[
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ]]
        return [np.asarray(final_bbox, dtype=np.float32)]
        
    def CSRT_tracker(self, frame):        
        if self.tracker_bbox == None and self.tracker_started == False:
            return
        
        if self.tracker_started == False:
            if self.tracker == None:
                if hasattr(cv2, "TrackerCSRT_create"):
                    self.tracker = cv2.TrackerCSRT_create()
                elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                    self.tracker = cv2.legacy.TrackerCSRT_create()
                else:
                    print("CSRT tracker unavailable; install opencv-contrib-python.")
                    self.tracker_bbox = None
                    return
        
        if self.tracker_bbox is not None and not self.tracker_started:
            try:
                self.start_time = time.time()
                self.tracker.init(frame, tuple(self.tracker_bbox))
                self.tracker_started = True
            except cv2.error as exc:
                print(f"tracker.init failed: {exc}")
                self.tracker = None
                return
        try:
            ok, updated_bbox = self.tracker.update(frame)
            if ok:
                self.tracker_bbox = list(updated_bbox)
        except:
            ok = None
            print("tracker.update failed")
        self.now_time = time.time()
        
        if self.now_time-self.start_time >= 2.0 :
            #cv2.putText(frame, "Please posture your hand correctly", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),1)
            cv2.putText(frame,'Posture your hand correctly',(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1, cv2.LINE_AA)
            #print("tracking timeout")
            self.tracker_started = False
            self.tracker_bbox = None
            return
            
        if ok:
            # Tracking success
            p1 = (int(self.tracker_bbox[0]), int(self.tracker_bbox[1]))
            p2 = (int(self.tracker_bbox[0] + self.tracker_bbox[2]), int(self.tracker_bbox[1] + self.tracker_bbox[3]))
            cv2.rectangle(frame, p1, p2, (80, 255, 255), 2, 1)
        else :
            # Tracking failure
            self.tracker_started = False
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            print("Tracking failure detected")
            #reintiallize code to tackle tracking failure
            
    
        
        
        

class Mouse:
    def __init__(self):
        self.tx_old = 0
        self.ty_old = 0
        self.trial = True
        self.flag = 0
        
    def move_mouse(self,frame,position,gesture):
        
        (sx,sy)=pyautogui.size()
        camy, camx = frame.shape[:2]
        (mx_old,my_old) = pyautogui.position()
        
        
        Damping = 2 # Hyperparameter we will have to adjust
        tx = position[0]
        ty = position[1]
        if self.trial:
            self.trial, self.tx_old, self.ty_old = False, tx, ty
        
        delta_tx = tx - self.tx_old
        delta_ty = ty - self.ty_old
        self.tx_old,self.ty_old = tx,ty
        
        if (gesture == 3):
            self.flag = 0
            mx = mx_old + (delta_tx*sx) // (camx*Damping)
            my = my_old + (delta_ty*sy) // (camy*Damping)            
            pyautogui.moveTo(mx,my, duration = 0.1)

        elif(gesture == 0):
            if self.flag == 0:
                pyautogui.doubleClick()
                self.flag = 1
        elif(gesture == 1):
            print('1 Finger Open')
        
        
        


class GestureController:
    gc_mode = 0
    pyautogui.FAILSAFE = False
    f_start_time = 0
    f_now_time = 0
    
    cam_width  = 0
    cam_height = 0
    
    aru_marker = None
    hand_roi = None
    glove = None
    csrt_track = None
    mouse = None
    
    def __init__(self):
        GestureController.aru_marker = Marker()
        GestureController.hand_roi = ROI(2.5, 2.5, 6, 0.45, 0.6, 0.4)
        GestureController.glove = Glove()
        GestureController.csrt_track = Tracker()
        GestureController.mouse = Mouse()
        GestureController.cap = cv2.VideoCapture(0)
        if GestureController.cap.isOpened():
            GestureController.cam_width  = int( GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
            GestureController.cam_height = int( GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
        else:
            print("CANNOT OPEN CAMERA")
        
        if not GestureController.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera index and permissions.")

        GestureController.gc_mode = 1
        GestureController.f_start_time = time.time()
        GestureController.f_now_time = time.time()
        
    def start(self):
        while (True):
            #mode checking
            if not GestureController.gc_mode:
                print('Exiting Gesture Controller')
                break
            #fps control
            fps = 30.0
            GestureController.f_start_time = time.time()
            while (GestureController.f_now_time-GestureController.f_start_time <= 1.0/fps):
                GestureController.f_now_time = time.time()
            
            #read camera
            ret, frame = GestureController.cap.read()
            if not ret or frame is None:
                print("Unable to read from camera. Check camera connection/permissions.")
                break
            frame = cv2.flip(frame, 1)
            
            #detect Marker, find ROI, find glove HSV, get FinalMask on glove
            GestureController.aru_marker.detect(frame)
            if GestureController.aru_marker.is_detected():
                GestureController.csrt_track.corners_to_tracker(GestureController.aru_marker.corners)
                GestureController.csrt_track.CSRT_tracker(frame)
                
            else:
                GestureController.csrt_track.tracker_bbox = None
                GestureController.csrt_track.CSRT_tracker(frame)
                GestureController.aru_marker.corners = GestureController.csrt_track.tracker_to_corner(GestureController.aru_marker.corners)
            
            FinalMask = None
            if GestureController.aru_marker.is_detected():
                GestureController.hand_roi.findROI(frame, GestureController.aru_marker)
                GestureController.hand_roi.find_glove_hsv(frame, GestureController.aru_marker)
                FinalMask = GestureController.hand_roi.cropROI(frame)
                GestureController.glove.find_fingers(FinalMask)
                GestureController.glove.find_gesture(frame)
                GestureController.mouse.move_mouse(frame,GestureController.hand_roi.marker_top,GestureController.glove.gesture)
            
            #draw call
            if GestureController.aru_marker.is_detected():
                GestureController.aru_marker.draw_marker(frame)
                draw_box(frame, GestureController.hand_roi.roi_corners, (255,0,0))
                draw_box(frame, GestureController.hand_roi.hsv_corners, (0,0,250))
                if FinalMask is not None:
                    cv2.imshow('FinalMask', FinalMask)
            
            #display frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # When everything done, release the capture
        GestureController.cap.release()
        cv2.destroyAllWindows()
        
        
        
