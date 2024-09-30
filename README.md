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

**Pose Estimation:** The first step to recognizing gestures can be to detect and localize human joints. Since body posture and hand placement are critical in my project, pose estimation techniques will be critical. I will use methods like OpenPose or MediaPipe to identify key body joints. 

**Correlation Filters:** Inspired by the work of David Bolme, correlation filters are an efficient method for gesture detection in drone footage. The idea is to learn kernels that maximize the response for specific patterns within an frame, highlighting regions that likely contain the gesture of interest. These filters can isolate important gestures, similar to the head detection example we were given in class but tailored for a unique set of drone control gestures.
