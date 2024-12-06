# Drone-Gesture-Recognition

CSE 60535: Computer Vision Semester Project

## Part 1: Conceptual design

### Project Overview
The goal of this project is to develop a computer vision-based model capable of recognizing human gestures and using this information to control a drone. Specifically, the drone should be able to detect gestures that signal the release of medical equipment at a specific location. I also plan to extend this model to allow for gesture-based drone navigation, which would enhance the drone's utility in emergency situations, such as search and rescue operations.

### Challenges
Several key challenges are anticipated in the development of this model:
- Recognizing human gestures accurately from drone footage, where the angles and perspectives may vary greatly, will be a significant challenge.
- While the MPII Human Pose Dataset provides a strong foundation, the poses captured in the dataset may not fully cover the gestures we require for drone control. Additionally, recognizing poses from aerial drone footage introduces variability in pose appearance due to changes in camera angle and altitude.
- The model must detect and recognize specific gestures. So we are not just building a model to detect a human, but also to be sensitive to specific gestures to control our drone. 

### Datasets
I plan to use a combination of existing datasets and our own custom drone data. 

My first dataset is the [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/) This dataset provides over 40K people with annotated body joints across 410 human activities. The data is well-suited for training models on general human pose estimation, especially for common gestures that may overlap with our use case. The dataset includes 2D annotations of body joints, and some of the images include occluded body parts and variations in lighting and angles, making it a robust starting point for our gesture recognition model. 

We plan to collect our own drone images to supplement the MPII dataset. This data will focus specifically on gestures relevant to drone control (e.g., hand signals for "release," "move forward," "hover"). By recording from the drone’s perspective at various altitudes and angles, we can create a more realistic test dataset. This test dataset will help evaluate the model's generalizability to aerial imagery, which is essential for our application.

I will primarily rely on the MPII dataset for training and validation. The drone footage that we collect will serve as our primary test set. This data will remain untouched until the final evaluation phase to ensure a fair assessment of the model's performance in real-world conditions.

### Solution

The high-level approach to solving this project involves a combination of pose estimation and gesture classification.

- **Pose Estimation:** The first step to recognizing gestures can be to detect and localize human joints. Since body posture and hand placement are critical in my project, pose estimation techniques will be critical. I will use methods like OpenPose or MediaPipe to identify key body joints. I will then put a classifier over this to classify the key points into the corresponding gestures from my gesture set.

- **Correlation Filters:** Inspired by the work of David Bolme, correlation filters can be used for sign recognition, in my case to mark where the drone will drop the medical equipment. The idea is to learn kernels that maximize the response for specific patterns within a frame, highlighting regions that likely contain the sign that we place on the ground to indicate where to drop the equipment. These filters can isolate important features, similar to the head detection example we were given in class but tailored for a unique sign that we can use in the real world.

## Part 2: Data acquisition and preparation

### Source of Data

- MPII Human Pose Dataset (train & validation set)
    - Download Link: [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/)
    - Associated papers: 2D Human Pose Estimation
        - New Benchmark and State of the Art Analysis by Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele.
        - Fine-grained Activity Recognition with Holistic and Pose based Features by Leonid Pishchulin, Mykhaylo Andriluka and Bernt Schiele.
    - This dataset contains over 40,000 annotated images across 410 activities, making it suitable for training a gesture recognition model.
- Custom Drone Footage (test set)
    - We plan to collect aerial drone footage featuring specific gestures needed for drone control: “release,” “move forward, backward, left, right” and “hover.” The footage will be recorded at various angles and altitudes to simulate real-world scenarios, making the dataset unique and tailored to this project.
    - Each image was extracted from YouTube videos. 

### Data Split Strategy 

- **Training Set:** From the MPII Human Pose Dataset, includes a wide range of human activities and poses in varied environments, lighting conditions, and body occlusions
- **Validation Set:** A subset of this data will be selected that closely resembles gestures similar to those needed for drone control. However, slight differences will be intentionally preserved to ensure the validation set is not too similar to the training set. This setup helps prevent overfitting and enables better generalization for this project.
- **Test Set:** We will capture multiple subjects (about 5-10 individuals) performing the drone control gestures across different sessions to create our test set.

## Part 3: First update

### Data Pre-Processing and Feature Extraction

#### Data Pre-Processing

- **YOLO Model for Person Detection:** The code utilizes a YOLO model to detect people in each frame. The model filters out detections based on confidence scores and person class IDs, helping to focus only on frames containing a person.
- **Cropping Detected Person:** Once a person is detected, the code extracts the region containing the person to simplify subsequent pose analysis and reduce background noise.
- **MediaPipe for Pose Estimation:** MediaPipe is used to process the cropped images, detecting key body landmarks (e.g., shoulders, elbows, wrists)

#### Feature Extraction

- **Landmark Extraction:** Specific body landmarks, such as shoulders, elbows, and wrists, are extracted to calculate joint angles and determine relative positions.
- **Angle Calculations:** The code computes angles between shoulders, elbows, and wrists to determine poses like the T-pose or touchdown pose.
- **Distance Normalization:** Shoulder width is calculated to normalize distances between key points, which is important for consistent detection across different scales.
- **Pose Detection Thresholding:** Horizontal and vertical alignment thresholds help distinguish different poses (e.g., T-pose, touchdown) based on arm positions.

### Why Did I Decide To Use These Algorithms

- **YOLO for Detection:** YOLO is a well-known object detection model that is efficient and accurate. Given that drone footage may have dynamic backgrounds and varying angles, YOLO’s ability to detect humans in these conditions is crucial. By narrowing down to high-confidence detections, it reduces false positives, focusing on frames with clear human presence.
- **MediaPipe for Pose Estimation:** MediaPipe is chosen for its real-time pose estimation capabilities. Unlike some other methods that may require significant computational resources, MediaPipe efficiently provides landmark detection and can handle variations in poses and orientations typical in drone footage.
- **Angle and Normalization-Based Pose Detection:** Calculating angles between joints allows for robust pose classification. Normalizing distances by shoulder width further improves pose detection across varying scales and distances, accommodating variations in altitude and angle specific to aerial footage.

### Demonstrating

https://drive.google.com/file/d/1EQn0DPFEtWbPe15KU32wNelTMklKSBkS/view

### Future Plans

In the current phase of the project, we are using a threshold-based classification approach ("brute force method") for pose detection, relying on calculated angles and normalized distances between key body landmarks. This method effectively distinguishes basic poses like the T-pose and touchdown pose by setting specific angle and distance thresholds. However, this approach is intended as a temporary solution, particularly for the upcoming demonstration of our lab’s drone capabilities in Oklahoma, starting on Nov 9. Using this method allows us to showcase the core functionality of the gesture recognition system with minimal computational requirements and robust, straightforward processing.

Looking ahead, I plan to transition the pose classification method to a neural network model. This upgrade will allow for more nuanced and accurate gesture recognition, as the network will be capable of learning and generalizing complex pose variations that are challenging to capture with hard-coded thresholds. After the demo, I will conduct a performance comparison between this initial threshold-based approach and the neural network model, assessing improvements in detection accuracy, robustness, and real-time processing efficiency, particularly within dynamic drone footage.

## Part 4: Second update

### Justification of the Choice of Classifier

The final solution I created two classifiers: a Support Vector Machine (SVM) and a Dense Neural Network (DNN). Both classifiers were chosen for their unique strengths and suitability for the specific requirements of the gesture recognition task.

#### Why SVM with RBF kernel? 

- nonlinearity
- SVC(kernel='rbf', gamma='scale', C=1.0)

#### Why a DNN

- Testing robustness
- 64 hidden units with ReLU activation

### Classification Accuracy
- SVM Classification Performance
- DNN Classification Performance
- videos of both

Both classifiers were validated on unseen drone footage in .mp4 format, and they correctly identified all gestures, further validating their generalization capabilities.

### Commentary on Accuracy and Ideas for Improvements

The observed perfect accuracies on both classifiers are promising but could indicate potential overfitting, especially considering the high dimensionality of the BODY25 keypoint data and the augmentation techniques used. The following considerations and improvements are proposed:

- Data Augmentation Validation:
    - While augmentation ensures robust training, it may inadvertently create synthetic data that is overly simplified or unrealistic. A thorough evaluation of the augmented data distribution compared to real-world drone footage could highlight discrepancies.
- Cross-validation:
    - Conduct k-fold cross-validation to evaluate model stability across various splits of training and validation data, providing a more robust assessment of generalization.
- Lightweight Model for Real-time Deployment:
    - The current DNN, while effective, could be optimized further for real-time deployment on drones with limited computational resources. Techniques such as model quantization or pruning could be considered.
    - Using Arturos yolo model. See if we is just using a bounding box. see if we can take the classes and train our data to get the gestures classifications…what he is using a class label.
 
### Implementation Notes
- Keypoint Conversion: Mediapipe output is converted to match OpenPose's BODY25 format to ensure compatibility with both the training data and real-time drone footage.
- Normalization: All keypoint coordinates are scaled to the range [0, 1] to account for variations in drone altitude and subject distance from the camera.
- Data Augmentation: The training dataset is augmented with transformations (scaling, rotation, translation, and noise addition) to increase robustness and balance class representation.
