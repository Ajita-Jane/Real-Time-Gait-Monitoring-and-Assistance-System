import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load trained model and scaler
model_path = "autoencoder_model.h5"
scaler_path = "scaler.pkl"

autoencoder = tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
scaler = joblib.load(scaler_path)

# Open Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully.")

def calculate_angle(a, b, c):
    """Calculate angle between three points: a -> b -> c"""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        extracted_features = []

        # Extract Key Points (X, Y, Z) for 12 joints
        for landmark_name in [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
        ]:
            landmark = landmarks[landmark_name]
            extracted_features.extend([landmark.x, landmark.y, landmark.z])

        # 🔥 ADDITIONAL FEATURES: Joint Angles (Theta values)
        try:
            left_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP])

            right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP])

            left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])

            right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])

            left_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                             landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE])

            right_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])

            extracted_features.extend([left_shoulder_angle, right_shoulder_angle,
                                       left_hip_angle, right_hip_angle,
                                       left_knee_angle, right_knee_angle])

        except:
            print("Skipping frame due to missing keypoints.")
            continue

        # 🔥 ADDITIONAL FEATURES: Gait Metrics (Dummy Values for Now)
        step_length = 0.5  # Placeholder (should be computed from ankle positions)
        step_width = 0.2   # Placeholder (should be computed from hip positions)
        feet_clearance = 0.1  # Placeholder
        left_stride_speed = 1.2  # Placeholder (should be computed from knee velocity)
        right_stride_speed = 1.3  # Placeholder

        extracted_features.extend([step_length, step_width, feet_clearance, left_stride_speed, right_stride_speed])

        # Ensure the extracted features match the expected count
        if len(extracted_features) != 47:
            print(f"Feature count mismatch: Expected 47, but got {len(extracted_features)}. Skipping frame.")
            continue

        # Normalize the extracted features
        extracted_features = np.array(extracted_features).reshape(1, -1)
        extracted_features = scaler.transform(extracted_features)

        # Predict Abnormal/Normal
        reconstructed = autoencoder.predict(extracted_features)
        error = np.mean(np.abs(extracted_features - reconstructed))
        status = "Normal Gait" if error < 6.07 else "Abnormal Gait"
        color = (0, 255, 0) if status == "Normal Gait" else (0, 0, 255)

        # Display status on frame
        cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Draw Skeleton
    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show Image in Window (For VS Code)
    cv2.imshow('Gait Analysis', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
