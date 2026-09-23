import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

class RouteConfidenceModel:
    """
    A lightweight ML model to classify route confidence without heavy neural networks.
    Target classes: 0 = valid route, 1 = uncertain route, 2 = junction candidate.
    """
    def __init__(self):
        self.model = LogisticRegression(max_iter=200)
        self._train_dummy_model()

    def _train_dummy_model(self):
        """
        Trains a baseline model on synthetic data to allow immediate prototyping.
        Features: [tape_area, centroid_variance, image_brightness, classical_confidence][cite: 1].
        """
        X_train = np.array([
            [5000, 15, 120, 0.9],  # Valid route (standard)
            [1500, 10, 180, 0.9],  # Valid route (bright room, thinner line)
            [1200, 5,  80,  0.9],  # Valid route (dark room, thinner line)
            [1000, 150, 80, 0.4],  # Uncertain route 
            [800,  200, 90, 0.3],  # Uncertain route
            [0,    0,   150, 0.0], # NO ROUTE 
            [300,  50,  125, 0.0], # NO ROUTE 
            [8500, 50, 130, 0.8],  # Junction candidate
            [9000, 60, 125, 0.85]  # Junction candidate
        ])
        y_train = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2]) # Updated labels to match
        self.model.fit(X_train, y_train)

    def predict(self, features):
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        return prediction, probabilities[prediction]


class VisionPipeline:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        
        # Configure ArUco Detector[cite: 1]
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Mock Camera Calibration for 3D Pose Estimation (Assumes 640x480 video)
        # Pose estimation requires intrinsic camera parameters[cite: 1].
        self.camera_matrix = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Define physical marker size (e.g., 0.05 meters = 5 cm)[cite: 1]
        self.marker_length = 0.05
        self.obj_points = np.array([
            [-self.marker_length/2,  self.marker_length/2, 0],
            [ self.marker_length/2,  self.marker_length/2, 0],
            [ self.marker_length/2, -self.marker_length/2, 0],
            [-self.marker_length/2, -self.marker_length/2, 0]
        ], dtype=np.float32)

        self.ml_model = RouteConfidenceModel()
        self.centroid_history = []

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        
        # 1. Detect ArUco Markers and Extract 3D Pose
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray_frame)
        
        marker_data = []
        if ids is not None:
            ids_flat = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i in range(len(ids_flat)):
                # Calculate 3D pose relative to the camera
                ret, rvec, tvec = cv2.solvePnP(self.obj_points, corners[i], self.camera_matrix, self.dist_coeffs)
                if ret:
                    # Draw X (red), Y (green), and Z (blue) axes extending from the marker
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
                
                marker_data.append({"id": int(ids_flat[i]), "corners": corners[i]})

        # 2. Crop to Floor Region[cite: 1]
        floor_roi = frame[height//2:height, :]
        roi_gray = cv2.cvtColor(floor_roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Thresholding for dark tape[cite: 1]
        # Blur aggressively to destroy carpet/floor texture noise
        blurred = cv2.GaussianBlur(roi_gray, (9, 9), 0)
        
        # Increase block size (31) and constant C (10) so only true dark lines trigger it
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        
        # 4. Find Contours and Heading Vector[cite: 1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tape_area = 0
        cx, cy = width // 2, height // 2
        classical_confidence = 0.0

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            tape_area = cv2.contourArea(largest_contour)
            
            # Raised the noise floor to 1500 pixels. 
            if tape_area > 1500:
                classical_confidence = 0.9
                
                # Draw the full detected boundary of the tape (in blue)
                contour_offset = largest_contour + np.array([0, height//2])
                cv2.drawContours(frame, [contour_offset], -1, (255, 0, 0), 2)

                # Calculate Centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"]) + (height // 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Calculate Heading Vector using Line of Best Fit
                line = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, x0, y0 = line[0][0], line[1][0], line[2][0], line[3][0]
                
                m = vy / (vx + 1e-5) 
                b = y0 - (m * x0)
                
                x1, x2 = 0, width
                y1 = int(m * x1 + b) + (height // 2)
                y2 = int(m * x2 + b) + (height // 2)
                
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 5. Extract ML Features & Predict Route Confidence
        self.centroid_history.append(cx)
        if len(self.centroid_history) > 5:
            self.centroid_history.pop(0)
        centroid_variance = np.var(self.centroid_history) if len(self.centroid_history) > 1 else 0

        image_brightness = np.mean(roi_gray)
        features = [tape_area, centroid_variance, image_brightness, classical_confidence]
        route_state, state_prob = self.ml_model.predict(features)

        state_labels = {0: "Valid Route", 1: "Uncertain", 2: "Junction Candidate"}
        classification = state_labels.get(route_state, "Unknown")

        cv2.putText(frame, f"State: {classification} ({state_prob:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"IDs: {[m['id'] for m in marker_data]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        
        lateral_error = (cx - (width / 2)) / (width / 2)
        return frame, lateral_error, classification, marker_data

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
                
            frame = cv2.resize(frame, (640, 480))
            
            # We need to extract the threshold image to display it
            height, width = frame.shape[:2]
            floor_roi = frame[height//2:height, :]
            roi_gray = cv2.cvtColor(floor_roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(roi_gray, (9, 9), 0)
            
            # Lowered the subtraction constant from 10 to 5 to be much more forgiving of room lighting
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)

            processed_frame, lateral_error, route_state, markers = self.process_frame(frame)
            
            cv2.imshow("Vision Pipeline POC", processed_frame)
            cv2.imshow("Debug: Threshold Mask", thresh) # Add this line to see the raw CV output
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    pipeline = VisionPipeline(source=0)
    pipeline.run()