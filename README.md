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
- *(More algorithms like Dense Optical Flow coming soon...)*

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
