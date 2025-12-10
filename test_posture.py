import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- CONFIGURATION (UPDATED FOR YOUR PATH) ---
# Use raw string (r"...") to handle backslashes in Windows paths safely
DATASET_ROOT = r"D:\computer vision\archive\data" 

# Define your class names exactly as the folders are named in 'data'
CLASSES = ["good", "bad"]

# Threshold: If neck bends more than this, it's "bad"
THRESHOLD_ANGLE = 15  

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    """ Calculates angle between three points. b is the vertex. """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def predict_posture(image_path):
    """
    Reads image, finds landmarks, predicts label based on geometry.
    Returns: 'good', 'bad', or 'unknown'
    """
    image = cv2.imread(image_path)
    if image is None:
        return "error_read"
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return "unknown" 

    landmarks = results.pose_landmarks.landmark
    h, w, _ = image.shape

    # Coordinates for Right Side (Ear, Shoulder)
    # 8: Ear, 12: Shoulder
    shoulder = [landmarks[12].x * w, landmarks[12].y * h]
    ear = [landmarks[8].x * w, landmarks[8].y * h]
    
    # Calculate Neck Inclination (Vertical -> Shoulder -> Ear)
    vertical_point = [shoulder[0], shoulder[1] - 100] 
    neck_inclination = calculate_angle(vertical_point, shoulder, ear)

    # Logic: High neck angle = Bad posture
    if neck_inclination > THRESHOLD_ANGLE:
        return "bad"
    else:
        return "good"

def main():
    print(f"--- Scanning Dataset at: {DATASET_ROOT} ---")
    
    y_true = []
    y_pred = []
    
    # Loop through the two main categories: 'good' and 'bad'
    for label in CLASSES:
        # Construct path: D:\...\data\good
        class_path = os.path.join(DATASET_ROOT, label)
        
        if not os.path.exists(class_path):
            print(f"Error: Path not found: {class_path}")
            continue

        print(f"Scanning folder: {label}...")
        
        # os.walk allows us to go into subfolders (like '1', '2', etc.)
        image_count = 0
        for root, dirs, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    
                    # Run prediction
                    prediction = predict_posture(full_path)
                    
                    if prediction in ["unknown", "error_read"]:
                        continue # Skip images where person isn't found
                    
                    y_true.append(label)
                    y_pred.append(prediction)
                    image_count += 1
                    
                    # Optional: Print progress every 50 images
                    if image_count % 50 == 0:
                        print(f"  Processed {image_count} images...")

        print(f"  -> Total valid images in '{label}': {image_count}")

    # --- RESULTS ---
    print("\n" + "="*40)
    print("       PERFORMANCE METRICS       ")
    print("="*40)
    
    if len(y_true) == 0:
        print("No valid images found! Check your paths.")
        return

    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.2f} ({acc*100:.2f}%)")
    
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    print("Confusion Matrix:")
    print(f"                 Predicted {CLASSES[0]}   Predicted {CLASSES[1]}")
    print(f"Actual {CLASSES[0]:<7}   {cm[0][0]:<15} {cm[0][1]}")
    print(f"Actual {CLASSES[1]:<7}   {cm[1][0]:<15} {cm[1][1]}")

if __name__ == "__main__":
    main()