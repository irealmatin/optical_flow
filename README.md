# Optical Flow Algorithms in OpenCV

A collection of Optical Flow algorithms and motion tracking implementations using Python and OpenCV. This repository aims to cover both Sparse and Dense optical flow techniques.

## 🧠 Concepts:

**Optical Flow**

The pattern of apparent motion of image objects between two consecutive frames. It assumes pixel intensity remains constant during motion.

**Lucas-Kanade Method (Sparse)**

A differential method that assumes the flow is essentially constant in a local neighborhood of the pixel (e.g., a 3x3 window). It solves the basic Optical Flow equation by using the Least Squares criterion.

- Feature Extraction: Uses Shi-Tomasi to find strong corners to track.

- Robustness: Uses Image Pyramids to handle large motions.

## 📂 Current Implementations

- **Sparse Optical Flow (Lucas-Kanade):** Tracks specific features (corners) using the Pyramid Lucas-Kanade method.
- **Optimized Sparse Optical Flow:** An enhanced version of the Lucas-Kanade method with forward-backward error checking and trajectory history.

## ⚙️ Prerequisites

Install the required dependencies:

```bash
pip install opencv-python numpy
```
## 🚀 How to Run
1. Clone the repository.
2. **IMPORTANT**: Open the python script (e.g., lucas_kanade.py) and change the video path in the __main__ section:
```python
# ⚠️ Change this to your video path or 0 for webcam
lucas_kanade_method("path/to/your/video.mp4")
```
 3. Run the script:
 
 ```bash
python lucas_kanade.py
```
or you can run the program by passing the video path as a command-line argument:
```bash
python demo.py --algorithm lucaskanade_interactive --video_path ../videos/car.mp4
``` 
or pass 0 to use the webcam:
```bash
python demo.py --algorithm lucaskanade --capture_index 0
```
4. Press 'q' to exit the video window.

notice that put the currect path to the video file in the code before running.
