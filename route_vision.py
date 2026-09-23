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
        Replace this with actual labeled data collected from track sessions.
        Features: [tape_area, centroid_variance, image_brightness, classical_confidence][cite: 1].
        """
        # Synthetic data: [area, variance, brightness, classical_score]
        X_train = np.array([
            [5000, 15, 120, 0.9],  # Valid route
            [5200, 10, 115, 0.95], # Valid route
            [1000, 150, 80, 0.4],  # Uncertain route (faded line/glare)
            [800,  200, 90, 0.3],  # Uncertain route
            [8500, 50, 130, 0.8],  # Junction candidate (wide area, stable centroid)
            [9000, 60, 125, 0.85]  # Junction candidate
        ])
        y_train = np.array([0, 0, 1, 1, 2, 2]) 
        self.model.fit(X_train, y_train)

    def predict(self, features):
        """
        Returns the predicted state and the probability of that state.
        """
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        return prediction, probabilities[prediction]


class VisionPipeline:
    def __init__(self, source=0):
        # Initialize video source (0 for default webcam, or path to an mp4 file)
        self.cap = cv2.VideoCapture(source)
        
        # Configure ArUco Detector for the specified 4x4 dictionary[cite: 1]
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        self.ml_model = RouteConfidenceModel()
        
        # State tracking for feature extraction
        self.centroid_history = []

    def process_frame(self, frame):
        """
        Executes the Sense -> Think pipeline for a single frame.
        """
        height, width = frame.shape[:2]
        
        # 1. Detect ArUco Markers first on the full color frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray_frame)
        
        marker_data = []
        if ids is not None:
            ids_flat = ids.flatten() # Safely converts to a 1D array
            for i in range(len(ids_flat)):
                marker_data.append({"id": int(ids_flat[i]), "corners": corners[i]})
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # 2. Crop to Floor Region (e.g., bottom half of the frame)[cite: 1]
        floor_roi = frame[height//2:height, :]
        roi_gray = cv2.cvtColor(floor_roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Thresholding for dark tape on a light floor[cite: 1]
        # Adjust block size and C value based on actual track lighting
        thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # 4. Find the Line Centroid[cite: 1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tape_area = 0
        cx, cy = width // 2, height // 2 # Default to center
        classical_confidence = 0.0

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            tape_area = cv2.contourArea(largest_contour)
            
            # Filter out noise (small blobs)[cite: 1]
            if tape_area > 500:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"]) + (height // 2) # Adjust for ROI offset
                    classical_confidence = 0.9 # High confidence if a large blob is found
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Update centroid history to calculate variance (for ML features)
        self.centroid_history.append(cx)
        if len(self.centroid_history) > 5:
            self.centroid_history.pop(0)
        centroid_variance = np.var(self.centroid_history) if len(self.centroid_history) > 1 else 0

        # 5. Extract ML Features & Predict Route Confidence
        image_brightness = np.mean(roi_gray)
        features = [tape_area, centroid_variance, image_brightness, classical_confidence]
        
        route_state, state_prob = self.ml_model.predict(features)

        # Map state integer back to string label
        state_labels = {0: "Valid Route", 1: "Uncertain", 2: "Junction Candidate"}
        classification = state_labels.get(route_state, "Unknown")

        # Overlay telemetry on frame for debugging
        cv2.putText(frame, f"State: {classification} ({state_prob:.2f})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"IDs: {[m['id'] for m in marker_data]}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        
        # Calculate lateral error (-1 to +1 normalized to frame width)[cite: 1]
        lateral_error = (cx - (width / 2)) / (width / 2)

        return frame, lateral_error, classification, marker_data

    def run(self):
        """
        Main execution loop.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Downsample to VGA or QVGA for processing speed if necessary[cite: 1]
            frame = cv2.resize(frame, (640, 480))
            
            processed_frame, lateral_error, route_state, markers = self.process_frame(frame)
            
            # Display output
            cv2.imshow("Vision Pipeline POC", processed_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start the pipeline using the default camera (0). 
    # Swap '0' with a video file path (e.g., 'test_run.mp4') to test recorded datasets.
    pipeline = VisionPipeline(source=0)
    pipeline.run()