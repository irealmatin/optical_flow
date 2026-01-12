import cv2 as cv 
import numpy as np

def dence_method(source):
    cap = cv.VideoCapture(source)
    ret , old_frame = cap.read()
    if not ret :
        return
    
    old_gray = cv.cvtColor(old_frame , cv.COLOR_BGR2GRAY)

    hsv = np.zeros_like(old_frame)
    hsv[...,1] = 255

    while True:
        ret , new_frame = cap.read()
        if not ret : 
            break

        new_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

        flow = cv.optflow.calcOpticalFlowSparseToDense(old_gray , new_gray , None , grid_step=8 , k=128)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        cv.imshow("Dense Optical Flow (Lucas-Kanade)", bgr)
        cv.imshow("Original", new_frame)

        if cv.waitKey(0) & 0xFF == ord('q'):
            break

        old_gray = new_gray

    cap.release()
    cv.destroyAllWindows()

def farneback_method(video_path):

    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret: return

    old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255 # Saturation 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


        flow = cv.calcOpticalFlowFarneback(
        old_gray, frame_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0 #scale_pyr - levels - winsize - iterations - n_poly - sigma_poly - flags
        ) 
        
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('Farneback Optical Flow', bgr)
        cv.imshow('Original', frame)

        if cv.waitKey(10) & 0xFF == ord('q'): 
            break
    old_gray = frame_gray

    cap.release()
    cv.destroyAllWindows()

if __name__=="__main__":
   # dence_method()
    farneback_method("videos/Highway_6436.mp4")
