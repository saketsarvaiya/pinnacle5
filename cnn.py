import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import json
import glob
import os
import matplotlib.pyplot as plt

def load_donkeycar_tub(tub_path):
    print(f"Scanning for DonkeyCar dataset in: {tub_path}...")
    X, y = [], []
    
    # 1. Search recursively for all .catalog files in nested tub folders
    catalog_files = glob.glob(os.path.join(tub_path, "**", "*.catalog"), recursive=True)
    
    if not catalog_files:
        print("CRITICAL: No .catalog files found! Double-check the folder path.")
        return np.array([]), np.array([])

    for catalog in catalog_files:
        catalog_dir = os.path.dirname(catalog) # Dynamically track the specific sub-tub
        
        with open(catalog, 'r') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                
                img_name = record.get("cam/image_array")
                steering = record.get("user/angle", 0.0)
                throttle = record.get("user/throttle", 0.0)
                
                if not img_name: continue
                
                # 2. Look for the image in the correct sub-tub directory
                img_path = os.path.join(catalog_dir, "images", img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # 3. Explicitly catch Git LFS failures 
                if img is None:
                    print(f"WARNING: Could not read {img_path}. If this happens a lot, Git LFS did not download the actual images.")
                    continue
                
                # Map Donkey steering logic to the required project classes[cite: 1]
                if throttle < 0.1:
                    label = 1 # Uncertain 
                elif abs(steering) > 0.6:
                    label = 2 # Junction Candidate 
                else:
                    label = 0 # Valid Route 
                    
                img = img / 255.0
                X.append(np.expand_dims(img, axis=-1))
                y.append(label)
                
    print(f"Successfully loaded {len(X)} images.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def generate_synthetic_track_data(samples=900):
    """
    Generates a plug-and-play dataset of 160x120 images to immediately test the CNN.
    Classes: 0 = Valid Route, 1 = Uncertain Route, 2 = Junction Candidate[cite: 1].
    """
    # 1 channel (grayscale) to reduce memory footprint and bandwidth[cite: 1].
    X = np.zeros((samples, 120, 160, 1), dtype=np.float32)
    y = np.zeros((samples,), dtype=np.int32)
    
    for i in range(samples):
        # Base floor color (light gray)
        img = np.ones((120, 160), dtype=np.float32) * 200 
        state = i % 3
        
        # Simulate different route geometries
        offset = int(np.random.randint(-30, 30))
        if state == 0:
            # Valid Route: Clear, continuous dark line
            cv2.line(img, (80, 120), (80 + offset, 0), 50, 15)
            y[i] = 0
        elif state == 1:
            # Uncertain Route: Faded or broken line (simulating glare/wear)
            cv2.line(img, (80, 120), (80 + offset, 60), 160, 15)
            y[i] = 1
        else:
            # Junction Candidate: Main line with a perpendicular branch
            cv2.line(img, (80, 120), (80, 0), 50, 15)
            cv2.line(img, (80, 60), (160, 60), 50, 15) 
            y[i] = 2
            
        # Add random camera noise to simulate hardware artifacts
        noise = np.random.normal(0, 15, (120, 160))
        img = np.clip(img + noise, 0, 255)
        
        # Normalize pixel values to 0.0 - 1.0 range
        X[i, :, :, 0] = img / 255.0
        
    return X, y

def build_delivery_vehicle_cnn():
    """
    A lightweight CNN optimized for the Pi 4 hardware footprint[cite: 1].
    """
    model = models.Sequential([
        # Input: 120x160 Grayscale image
        layers.Input(shape=(120, 160, 1)),
        
        # Conv Block 1: Feature extraction (edges, lines)
        layers.Conv2D(16, (3, 3), activation='relu', strides=(2, 2)),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 2: Higher-level geometries (branches)
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 3: Compressed spatial features
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier Head
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3), # Prevent overfitting on small track datasets
        
        # Output: 3 route state probabilities
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def visualize_predictions(model, X_test, y_test, num_samples=9):
    """
    Displays a 3x3 grid of random test images, comparing the CNN's prediction 
    against the actual human-driven label.
    """
    print(f"\nGenerating visualization for {num_samples} random test samples...")
    
    # Pick random indices from the test set
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    class_names = ["Valid Route", "Uncertain", "Junction"]
    
    plt.figure(figsize=(12, 10))
    
    for i, idx in enumerate(indices):
        img = X_test[idx]
        true_label = y_test[idx]
        
        # Run inference on the single image
        pred_probs = model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_label = np.argmax(pred_probs)
        confidence = np.max(pred_probs)
        
        # Set up the subplot
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(img.squeeze(), cmap='gray') # squeeze removes the channel dimension for plotting
        
        # Green text for correct predictions, Red for incorrect
        color = 'green' if true_label == pred_label else 'red'
        title_text = f"Pred: {class_names[pred_label]} ({confidence:.2f})\nTrue: {class_names[true_label]}"
        
        plt.title(title_text, color=color, fontsize=10)
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X_data, y_data = load_donkeycar_tub(r"C:\Users\_____\dataset\donkey_datasets\circuit_launch_20210716\murmurpi4_circuit_launch_20210716_1611")
    
    # 4. Halt immediately if the array is empty
    if len(X_data) == 0:
        raise ValueError("Dataset is empty. Ensure Git LFS pulled the actual .jpg files and the path is correct.")
    
    # Session-separated split logic[cite: 1]
    split_idx = int(len(X_data) * 0.8)
    X_train, X_test = X_data[:split_idx], X_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    
    print("Building lightweight CNN model...")
    model = build_delivery_vehicle_cnn()
    
    print("\nTraining CNN...")
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32)
    
    print("\nTest inference on a single frame:")
    sample_frame = np.expand_dims(X_test[0], axis=0) # Add batch dimension for inference
    predictions = model.predict(sample_frame)
    predicted_class = np.argmax(predictions)
    
    class_names = ["Valid Route", "Uncertain Route", "Junction Candidate"]
    print(f"Predicted State: {class_names[predicted_class]} (Confidence: {np.max(predictions):.2f})")

    visualize_predictions(model, X_test, y_test)