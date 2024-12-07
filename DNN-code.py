
import json
import glob
import numpy as np
import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ==========================================
# Load Data
# ==========================================

openpose_json_files = glob.glob('Data/Points/*.json')

openpose_data = []
openpose_labels = []

def get_label_from_filename(filename):
    filename_lower = filename.lower()
    if 't_body' in filename_lower:
        return 'T-pose'
    elif 'stand_body' in filename_lower:
        return 'Stand'
    elif 'leftarm' in filename_lower:
        return 'Left Arm Raised'
    elif 'rightarm' in filename_lower:
        return 'Right Arm Raised'
    elif 'traffic_allstop' in filename_lower:
        return 'Traffic All Stop'
    else:
        return 'Unknown'

for file in openpose_json_files:
    with open(file, 'r') as f:
        content = json.load(f)
        openpose_data.append(content)
        label = get_label_from_filename(file)
        openpose_labels.append(label)

# ==========================================
# Data Augmentation Functions
# ==========================================

def scale_keypoints(keypoints, scale_factor):
    keypoints_scaled = keypoints * scale_factor
    keypoints_scaled = np.clip(keypoints_scaled, 0.0, 1.0)
    return keypoints_scaled

def rotate_keypoints(keypoints, angle):
    angle_rad = np.deg2rad(angle)
    x = keypoints[:25]
    y = keypoints[25:]
    x_center = np.mean(x)
    y_center = np.mean(y)
    x = x - x_center
    y = y - y_center
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    x_rot = x_rot + x_center
    y_rot = y_rot + y_center
    keypoints_rotated = np.concatenate([x_rot, y_rot])
    keypoints_rotated = np.clip(keypoints_rotated, 0.0, 1.0)
    return keypoints_rotated

def translate_keypoints(keypoints, translation):
    tx, ty = translation
    x = keypoints[:25] + tx
    y = keypoints[25:] + ty
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    keypoints_translated = np.concatenate([x, y])
    return keypoints_translated

def add_noise_keypoints(keypoints, noise_level):
    noise = np.random.normal(0, noise_level, size=keypoints.shape)
    keypoints_noisy = keypoints + noise
    keypoints_noisy = np.clip(keypoints_noisy, 0.0, 1.0)
    return keypoints_noisy

# ==========================================
# Prepare Features and Labels with Data Augmentation
# ==========================================

features = []
labels = []

for i, entry in enumerate(openpose_data):
    label = openpose_labels[i]
    print(label)
    for frame in entry['data']:
        x = np.array(frame['x'])
        y = np.array(frame['y'])

        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        if x_max - x_min != 0:
            x_norm = (x - x_min) / (x_max - x_min)
        else:
            x_norm = x

        if y_max - y_min != 0:
            y_norm = (y - y_min) / (y_max - y_min)
        else:
            y_norm = y

        keypoints_norm = np.concatenate([x_norm, y_norm])
        features.append(keypoints_norm)
        labels.append(label)

        # Augment
        # Scales
        scales = [0.9, 1.1]
        for scale in scales:
            features.append(scale_keypoints(keypoints_norm, scale_factor=scale))
            labels.append(label)

        # Rotation
        angles = [-15, 15]
        for angle in angles:
            features.append(rotate_keypoints(keypoints_norm, angle=angle))
            labels.append(label)

        # Translation
        translations = [(-0.05, 0), (0.05, 0), (0, -0.05), (0, 0.05)]
        for translation in translations:
            features.append(translate_keypoints(keypoints_norm, translation=translation))
            labels.append(label)

        # Noise
        noise_levels = [0.02]
        for noise_level in noise_levels:
            features.append(add_noise_keypoints(keypoints_norm, noise_level=noise_level))
            labels.append(label)

X = np.array(features)
y = np.array(labels)

print(f"Feature matrix shape after augmentation: {X.shape}")
print(f"Labels shape after augmentation: {y.shape}")

# ==========================================
# Split and Encode
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

num_classes = len(np.unique(y_train_encoded))
y_train_cat = to_categorical(y_train_encoded, num_classes=num_classes)
y_test_cat = to_categorical(y_test_encoded, num_classes=num_classes)

with open('dnn_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# ==========================================
# MediaPipe Keypoint Extraction and Prediction Functions
# ==========================================

MEDIAPIPE_TO_OPENPOSE_INDICES = [
    0,   # NOSE
    None,# NECK (does not exist in MediaPipe)
    11,  # RIGHT_SHOULDER
    13,  # RIGHT_ELBOW
    15,  # RIGHT_WRIST
    12,  # LEFT_SHOULDER
    14,  # LEFT_ELBOW
    16,  # LEFT_WRIST
    None,# MID_HIP (does not exist in MediaPipe)
    23,  # RIGHT_HIP
    25,  # RIGHT_KNEE
    27,  # RIGHT_ANKLE
    24,  # LEFT_HIP
    26,  # LEFT_KNEE
    28,  # LEFT_ANKLE
    1,   # RIGHT_EYE
    2,   # LEFT_EYE
    3,   # RIGHT_EAR
    4,   # LEFT_EAR
    31,  # LEFT_FOOT_INDEX
    31,  # duplicate
    29,  # LEFT_HEEL
    32,  # RIGHT_FOOT_INDEX
    32,  # duplicate
    30,  # RIGHT_HEEL
]

def normalize_coordinates(x_coords, y_coords):
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    if (x_max - x_min) != 0:
        x_norm = (x_coords - x_min) / (x_max - x_min)
    else:
        x_norm = x_coords
    if (y_max - y_min) != 0:
        y_norm = (y_coords - y_min) / (y_max - y_min)
    else:
        y_norm = y_coords

    # Flip the axes
    x_norm_flipped = 1 - x_norm
    y_norm_flipped = 1 - y_norm

    return x_norm_flipped.tolist(), y_norm_flipped.tolist()

def extract_mediapipe_keypoints(results_mp, frame_width, frame_height):
    if not results_mp.pose_landmarks:
        return None

    x_coords = []
    y_coords = []

    for idx in MEDIAPIPE_TO_OPENPOSE_INDICES:
        if idx is not None and idx < len(results_mp.pose_landmarks.landmark):
            landmark = results_mp.pose_landmarks.landmark[idx]
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
        else:
            x_coords.append(0.5)
            y_coords.append(0.5)

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    x_coords, y_coords = normalize_coordinates(x_coords, y_coords)
    return x_coords + y_coords

def predict_gesture_dnn(model, coordinates_array):
    keypoints_array = np.array(coordinates_array).reshape(1, -1)
    preds = model.predict(keypoints_array, verbose=0)
    predicted_class_index = np.argmax(preds, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class_index)
    return predicted_label[0]

def evaluate_on_4_images(model, test_images):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    correct_count = 0

    for img_path, expected_label in test_images:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read {img_path}. Skipping.")
            continue
        frame_height, frame_width, _ = image.shape
        results_mp = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results_mp.pose_landmarks:
            print("No landmarks detected for", img_path)
            continue
        coordinates_array = extract_mediapipe_keypoints(results_mp, frame_width, frame_height)
        if coordinates_array is None:
            print("No keypoints detected for", img_path)
            continue
        predicted_label = predict_gesture_dnn(model, coordinates_array)
        if predicted_label == expected_label:
            correct_count += 1

    return correct_count

# ==========================================
# Define Model Creation Function
# ==========================================

def create_model(num_hidden_layers=1, units=64, learning_rate=1e-3):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(50,)))
    for _ in range(num_hidden_layers):
        model.add(keras.layers.Dense(units, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================================
# Train a Single Model
# ==========================================

# Set hyperparameters here
num_hidden_layers = 2
units = 64
learning_rate = 1e-3
epochs = 10

model = create_model(num_hidden_layers=num_hidden_layers, units=units, learning_rate=learning_rate)
model.fit(X_train, y_train_cat, epochs=epochs, batch_size=32, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# Generate Classification Report and Confusion Matrix
# ==========================================

# Get predictions for the test set
y_test_pred = model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)  # Convert probabilities to class indices

# Decode class indices to labels
y_test_true_labels = label_encoder.inverse_transform(np.argmax(y_test_cat, axis=1))
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred_classes)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_true_labels, y_test_pred_labels, target_names=label_encoder.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_true_labels, y_test_pred_labels, labels=label_encoder.classes_)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Paths and expected labels for the 4 test images
test_images = [
    ("/Users/demetriushernandez/Desktop/t-rotate.png", "T-pose"),
    ("/Users/demetriushernandez/Desktop/touchdown-rotate.png", "Traffic All Stop"),
    ("/Users/demetriushernandez/Desktop/t-rotate-close.png", "T-pose"),
    ("/Users/demetriushernandez/Desktop/touchdown-rotate-close.png", "Traffic All Stop"),
]

correct_count = evaluate_on_4_images(model, test_images)
print(f"Correct classification on the 4 images: {correct_count}/4")

# Save the model
model.save('model.h5')
print("Model saved to model.h5")

from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Load the saved model
model = load_model('model.h5')
model.summary()

# Attempt to print optimizer details (learning rate)
if hasattr(model, 'optimizer'):
    lr = model.optimizer.learning_rate.numpy()
    print(f"Learning Rate: {lr}")
else:
    print("Could not determine learning rate from model.")

# Load the label encoder
with open('dnn_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)
print(f"Classes: {label_encoder.classes_}")

# ==========================================
# Testing the Neural Network Model on MP4
# ==========================================

yolo_model = YOLO('30to50.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

video_path = 'Data/Test Data/h-30-meters.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = 'output_with_gestures.mp4'
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = yolo_model.predict(img_rgb, imgsz=640, conf=0.5)
    detections = results[0]

    for detection in detections.boxes:
        if int(detection.cls[0]) == 0:  # 'person' class
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            box_width = x2 - x1
            box_height = y2 - y1
            expansion_ratio = 0.5
            x_expansion = int(box_width * expansion_ratio / 2)
            y_expansion = int(box_height * expansion_ratio / 2)
            x1_expanded = max(0, x1 - x_expansion)
            y1_expanded = max(0, y1 - y_expansion)
            x2_expanded = min(frame_width, x2 + x_expansion)
            y2_expanded = min(frame_height, y2 + y_expansion)

            cv2.rectangle(
                frame,
                (x1_expanded, y1_expanded),
                (x2_expanded, y2_expanded),
                (0, 255, 0),
                2
            )

            person_img = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

            if person_img.size == 0:
                continue

            person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            results_mp = pose.process(person_img_rgb)

            if results_mp.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded],
                    results_mp.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                coordinates_array = extract_mediapipe_keypoints(results_mp, frame_width, frame_height)
                if coordinates_array is None:
                    continue

                predicted_label = predict_gesture_dnn(model, coordinates_array)

                cv2.putText(
                    frame,
                    predicted_label,
                    (x1_expanded, y1_expanded - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    'No Pose Detected',
                    (x1_expanded, y1_expanded - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
