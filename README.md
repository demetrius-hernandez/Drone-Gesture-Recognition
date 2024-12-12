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

In this section, I present the completed solution along with classification experiments, justification for the chosen classifiers, evaluation metrics on both training and validation sets, and ideas for future improvements.

At this stage, I incorporated a new dataset with labeled keypoints for training. This decision was made upon realizing that the original dataset lacked the necessary features to effectively train a robust and accurate model for the intended application [New Dataset](https://github.com/ArthurFDLR/pose-classification-kit/tree/master/pose_classification_kit/datasets/Body). The entire dataset consists of 20 body classes, each containing between 500 and 600 samples, resulting in a total of 10,680 entries. Each entry is represented as an array of 25 2D coordinates, mapped according to the BODY25 keypoint model. For this project, I am using a subset of the total dataset to focus on five specific gestures: Stand, All Stop (Touchdown Pose), T-pose, Right Arm Up, and Left Arm Up. 

### Justification of the Choice of Classifier

For this stage of the project, I experimented with two classifiers to handle the gesture recognition task: a Support Vector Machine (SVM) with an RBF kernel and a Dense Neural Network (DNN).

- SVM with RBF Kernel
    - I chose an SVM with an RBF kernel because the BODY25 keypoint data is essentially a set of coordinates and potentially complex pose patterns. An SVM with an RBF kernel can model these complex relationships without requiring a large amount of training data. It also tends to provide good generalization and can be more robust to small datasets or limited feature sets.

- Dense Neural Network (DNN)
    - The DNN was chosen to test the robustness and generalization capabilities of a neural model. Neural networks can learn complex, non-linear mappings from inputs (keypoint coordinates) to class labels (gestures). By using several dense layers with ReLU activations, the DNN can capture subtle variations in human poses. This approach is easily scalable for additional classes and can be fine-tuned if new gestures or more complex scenarios are introduced.
 
Together, these classifiers serve as complementary approaches—SVM offers a strong, well-understood baseline, while the DNN provides the flexibility and scalability that might be needed for more complex real-world scenarios under a black-box.

### Classification Accuracy and Evaluation Metrics

For this project, I used a combination of metrics to evaluate the performance:

- **Precision, Recall, and F1-Score:** These metrics provide a detailed view of classification performance on each gesture class, showing how often gestures are correctly identified and how often the model confuses one gesture with another.
- **Confusion Matrix:** Helps visualize the distribution of predictions across classes.
- **Accuracy:** Gives a quick summary of overall performance.

#### Performance on Training and Validation Subsets


##### SVM with RBF Kernel:

Example of the SVM on real drone data: [https://drive.google.com/file/d/1TzSEVmxbjz-hQy4rHJIxNSJCbq0ZOnCp/view?usp=sharing](https://drive.google.com/file/d/1TzSEVmxbjz-hQy4rHJIxNSJCbq0ZOnCp/view?usp=sharing)

Training accuracy also reached ~100%, suggesting that the SVM was able to perfectly fit the training set as well.

The Classification Report for the SVM:

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Left Arm Raised    | 1.00      | 1.00   | 1.00     | 1068    |
| Right Arm Raised   | 1.00      | 1.00   | 1.00     | 1006    |
| Stand              | 1.00      | 1.00   | 1.00     | 1042    |
| T-pose             | 1.00      | 1.00   | 1.00     | 1006    |
| Traffic All Stop   | 1.00      | 1.00   | 1.00     | 1142    |
|                    |           |        |          |         |
| **Accuracy**       |   1.00    |        |          |   5264  |
| **Macro Avg**      | 1.00      | 1.00   | 1.00     | 5264    |
| **Weighted Avg**   | 1.00      | 1.00   | 1.00     | 5264    |

The confusion matrix for the SVM classifier:

|                     | Left Arm Raised | Right Arm Raised | Stand | T-pose | Traffic All Stop |
|----------------------|-----------------|------------------|-------|--------|------------------|
|   Left Arm Raised    | 1068           | 0                | 0     | 0      | 0                |
|   Right Arm Raised   | 0              | 1005             | 1     | 0      | 0                |
|   Stand              | 0              | 0                | 1042  | 0      | 0                |
|   T-pose             | 0              | 0                | 0     | 1006   | 0                |
|   Traffic All Stop   | 0              | 0                | 0     | 0      | 1142             |

The confusion matrix shows almost no misclassifications on the data, indicating a ~100% accuracy.

##### Dense Neural Network:

Example of the DNN on real drone data: [https://drive.google.com/file/d/1ptyfryyE0CmfNhRG2dsAQmgExcALm061/view?usp=sharing](https://drive.google.com/file/d/1ptyfryyE0CmfNhRG2dsAQmgExcALm061/view?usp=sharing)

The DNN also achieved ~100% accuracy during training, indicating that it effectively learned the training data patterns.

The Classification Report for the DNN:

| Class               | Precision | Recall | F1-Score | Support |
|---------------------|-----------|--------|----------|---------|
| Left Arm Raised     | 1.00      | 1.00   | 1.00     | 1068    |
| Right Arm Raised    | 1.00      | 1.00   | 1.00     | 1006    |
| Stand               | 1.00      | 1.00   | 1.00     | 1042    |
| T-pose              | 1.00      | 1.00   | 1.00     | 1006    |
| Traffic All Stop    | 1.00      | 1.00   | 1.00     | 1142    |
|                     |           |        |          |         |
| **Accuracy**        |           |        | 1.00     | 5264    |
| **Macro Avg**       | 1.00      | 1.00   | 1.00     | 5264    |
| **Weighted Avg**    | 1.00      | 1.00   | 1.00     | 5264    |


The confusion matrix for the DNN classifier:

|                 | Left Arm Raised | Right Arm Raised | Stand | T-pose | Traffic All Stop |
|-----------------|-----------------|------------------|-------|--------|------------------|
| Left Arm Raised | 1068            | 0                | 0     | 0      | 0                |
| Right Arm Raised| 0               | 1004             | 2     | 0      | 0                |
| Stand           | 0               | 1                | 1041  | 0      | 0                |
| T-pose          | 0               | 0                | 0     | 1006   | 0                |
| Traffic All Stop| 0               | 0                | 0     | 0      | 1142             |

The confusion matrix shows almost no misclassifications on the data, indicating a ~100% accuracy.

### Commentary on Accuracy and Ideas for Improvements

While the near perfect accuracy scores are encouraging, they raise the question of overfitting. Such results suggest that either:

- The dataset might not fully represent the complexity of real-world drone footage, and our models are overfitting to this controlled scenarios.
- The feature representation (BODY25 keypoints plus normalization) may be too "clean," making the classification task simpler than expected.
- Our augmentation strategies or data splits may not accurately reflect the complexities of the intended real-world environment.
-  **Performance on Real Drone Footage:**
I tested the model on the actual drone-collected footage, which had been held out during training, to evaluate its accuracy in real-world conditions. A drop in accuracy would have indicated a potential gap between the training/validation scenarios and the real-world application, requiring adjustments to the model architecture or the addition of more drone-collected training samples. However, in this case, the model maintained high accuracy on the drone footage. Every frame, as assessed through a manual review, was classified correctly, suggesting that the model generalizes well to real-world scenarios.

#### Proposed Improvements:

- **More Realistic Validation Data:**
Introduce more challenging validation sets with varied lighting, subjects wearing different clothing, and background clutter. This would help ensure that the classifier generalizes beyond the current dataset.

- **Cross-Validation:**
Implement k-fold cross-validation to assess stability and generalization. If performance drops on different folds, it will highlight overfitting.

- **Lightweight Models:**
For real-time drone deployment, consider making the models smaller (pruning) or techniques to run inference efficiently on limited hardware.

- Explore Arturo's (my labmate) YOLO model to determine if it primarily uses bounding boxes for object detection. Investigate whether the model's class labels can be adapted or extended to train and classify specific gestures for our dataset. This approach would allow us to only have to deploy the existing yolo model on our drones. 
 
### Implementation Notes
- Keypoint Conversion: Mediapipe output is converted to match OpenPose's BODY25 format to ensure compatibility with the drone footage.
- Normalization: All keypoint coordinates are scaled to the range [0, 1]
- Data Augmentation: The training dataset is augmented with transformations (scaling, rotation, translation, and noise addition) to increase robustness and balance class representation.

## Part 5: Final Update (in progress)

### Description of Unseen Data

For the final evaluation, I collected a test dataset comprising drone footage of individuals performing a sequence of predefined gestures. This footage was captured using a drone camera, introducing real-world challenges such as variations in lighting, angles, and altitudes (although I was careful to limit the variance caused by these variables). The set consists of different gesture sequences, with each sequence lasting between 4-6 seconds and containing combinations of gestures that correspond to specific drone commands.

### Differences from Training/Test/Validation Sets

- **Perspective:** The training/test/validation datasets primarily consisted of keypoints derived from static, ground-level images. In contrast, this unseen dataset features aerial drone footage, which introduces perspective distortion and changes in the appearance of gestures.

### Observed Overfitting and Implications

The results indicated minimal differences between the training/validation performance and the unseen dataset accuracy. This consistency suggests that while the model is likely overfitting to the training data, the poses in this dataset are distinct enough for the overfitting to have little practical effect. However, this may not hold as the complexity of gestures increases. For example:

- **Similar Gestures:** If future gestures have subtler distinctions (e.g., slight hand movements), overfitting to specific angles or body positions could lead to misclassification.
- **Occluded Points:** As occlusions increase in more complex poses, the model might struggle without additional training data or more advanced augmentation strategies.

### Observed Errors
- **Ambiguous Gestures:** Some misclassifications occurred due to gestures that were not part of the training set but resembled existing classes. For instance, when a subject raised both hands straight up, the model classified it as Traffic All Stop.

### Why This Approach is Sufficietly Tested (but can be improved)

I believe the final programs are sufficiently tested because they were evaluated on a separate, unseen dataset of real-world drone footage. This dataset simulates the intended deployment scenario by introducing natural variations in environment, lighting, and perspective. Moreover, the test data has been carefully crafted to include a representative range of gestures relevant to drone control. By keeping the test data entirely isolated during training, I ensured an unbiased evaluation of the model's performance, providing confidence that the results reflect its robustness and real-world applicability.

However, the model can be improved a lot, which you can see in the following point. 

### Proposed Improvements (before we actually put this on our drones)

- **Adding an Unknown Gesture Class:** Introduce an “unknown” gesture class to handle cases where the input does not match any predefined gesture. This would improve robustness without significantly reducing overall performance.
- **Enhanced Training Data:** Include additional samples in the training set with: Simulated occlusions and lighting variations, gestures performed at varying angles and distances from the camera.
- **Model Architecture Adjustments:** Fine-tune the YOLO model to detect not just bounding boxes but also specific gesture-related features.
- **Post-Processing:** Implement a confidence threshold for classification. For gestures with low confidence, classify them as “unknown” or request re-performance.

### Presentation Materials

- **Pipeline Overview:**: I use a YOLO model, trained specifically for detecting people from drones, which I obtained from Arturo, my labmate. The YOLO model identifies bounding boxes around individuals in the drone footage. For each detected bounding box, I extract the corresponding region of the frame and pass it to MediaPipe for keypoint detection. MediaPipe outputs keypoints in the Body33 format, which I then convert to the Body25 format to match the input structure required by my classifier. Finally, I feed the transformed 25 keypoints into the classifier, which labels the frame based on the identified gesture.

- To demonstrate the project’s functionality, I prepared a short video showcasing the model’s performance on the unseen test set. The video demonstration is available here:

[https://drive.google.com/file/d/1YGL4H4ETyiUt7bXlNK8EGFDmGjgHk2sy/view?usp=sharing](https://drive.google.com/file/d/1YGL4H4ETyiUt7bXlNK8EGFDmGjgHk2sy/view?usp=sharing)


In the video, the individual begins by holding a Touchdown pose for 3 seconds, followed by a T-pose for 3 seconds, which signals the drop of medical equipment. Next, the individual raises their left hand for 3 seconds and their right hand for 3 seconds, representing the command to fly upward. These commands are basic examples and will be enhanced with additional gestures in the future. This demonstration showcases the current capabilities for this class. Note that the animations in the video were manually performed for demonstration purposes.
