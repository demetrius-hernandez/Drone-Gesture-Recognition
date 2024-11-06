import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Load your custom detection model
model_det = YOLO("30to50.pt")  # Replace with the correct path to your custom detection model

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the pose estimator
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Open the video file
# cap = cv2.VideoCapture('Flight Data/30-straight-ahead-grass.MP4')  # Replace with your MP4 video file path
cap = cv2.VideoCapture('Flight Data/example-2.MP4')



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video file")
        break

    # Run detection on the frame with verbose=False to suppress messages
    results = model_det.predict(frame, imgsz=640, verbose=False)

    # Get detections
    detections = results[0]
    boxes = detections.boxes

    for box in boxes:
        cls_id = int(box.cls[0])  # Class ID
        conf = float(box.conf[0])  # Confidence score

        # Assuming class ID 0 is 'person' # ---------- adjust if necessary ---------- #
        if cls_id == 0 and conf > 0.5:
            x1, y1, x2, y2 = box.xyxy[0]

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Crop the person from the frame
            person_img = frame[y1:y2, x1:x2]

            # Check if the cropped image is valid
            if person_img.size == 0:
                continue

            # Convert the image to RGB
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe
            results_pose = pose.process(person_rgb)

            if results_pose.pose_landmarks:
                # Draw pose landmarks on the cropped person image (optional)
                mp_drawing.draw_landmarks(
                    person_img, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Get landmarks
                landmarks = results_pose.pose_landmarks.landmark

                # Image dimensions
                image_height, image_width, _ = person_img.shape

                # Function to calculate angle between three points
                def calculate_angle(a, b, c):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)

                    ba = a - b
                    bc = c - b

                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.degrees(np.arccos(cosine_angle))
                    return angle

                # Extract required landmarks and convert to pixel coordinates
                def get_landmark_coords(landmark):
                    return [landmark.x * image_width, landmark.y * image_height]

                left_shoulder = get_landmark_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                right_shoulder = get_landmark_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                left_elbow = get_landmark_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
                right_elbow = get_landmark_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
                left_wrist = get_landmark_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
                right_wrist = get_landmark_coords(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

                # Calculate angles at elbows
                left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Calculate shoulder width to normalize distances
                shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))

                # Calculate horizontal distances between wrists and shoulders
                left_wrist_shoulder_dx = abs(left_wrist[0] - left_shoulder[0])
                right_wrist_shoulder_dx = abs(right_wrist[0] - right_shoulder[0])

                # Normalize horizontal distances by shoulder width
                left_wrist_shoulder_dx_norm = left_wrist_shoulder_dx / shoulder_width
                right_wrist_shoulder_dx_norm = right_wrist_shoulder_dx / shoulder_width

                # Calculate vertical distances between wrists and shoulders
                left_wrist_shoulder_dy = abs(left_wrist[1] - left_shoulder[1])
                right_wrist_shoulder_dy = abs(right_wrist[1] - right_shoulder[1])

                # Normalize vertical distances by shoulder width
                left_wrist_shoulder_dy_norm = left_wrist_shoulder_dy / shoulder_width
                right_wrist_shoulder_dy_norm = right_wrist_shoulder_dy / shoulder_width

                # Set thresholds (adjust these values based on testing)
                horizontal_extension_threshold = 0.4
                vertical_alignment_threshold = 0.3  # For T-pose, wrists should not be too high or low compared to shoulders

                # Check if arms are horizontally extended
                left_arm_extended = left_wrist_shoulder_dx_norm > horizontal_extension_threshold
                right_arm_extended = right_wrist_shoulder_dx_norm > horizontal_extension_threshold

                # Check if wrists are at similar vertical level as shoulders
                left_arm_aligned = left_wrist_shoulder_dy_norm < vertical_alignment_threshold
                right_arm_aligned = right_wrist_shoulder_dy_norm < vertical_alignment_threshold

                # Determine the pose based on arm angles, horizontal extension, and vertical alignment
                pose_label = 'Other Pose'

                # Check for T-pose
                if (160 < left_arm_angle < 200 and 160 < right_arm_angle < 200 and
                    left_arm_extended and right_arm_extended and
                    left_arm_aligned and right_arm_aligned):
                    pose_label = 'T-Pose'
                # Check for Touchdown pose
                elif (70 < left_arm_angle < 110 and 70 < right_arm_angle < 110 and
                      left_arm_extended and right_arm_extended and
                      left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]):
                    # For touchdown pose, wrists should be above shoulders
                    pose_label = 'Touchdown Pose'

                # Display the pose label on the original frame
                cv2.putText(frame, pose_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2, cv2.LINE_AA)

                # Draw rectangle around person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            else:
                # No pose landmarks detected
                cv2.putText(frame, 'No Pose Detected', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw rectangle around person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()