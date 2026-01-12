import cv2 as cv
import numpy as np 

def rlof_method(source):

    cap = cv.VideoCapture(source)
    ret , frame = cap.read()
    if not ret :
        return
    
    old_frame = frame 

    hsv = np.zeros_like(frame)
    hsv[...,1] = 255

    while True:
        ret , frame = cap.read()
        if not ret :
            return
        
        flow = cv.optflow.calcOpticalFlowDenseRLOF(old_frame , frame , None)

        mag , ang = cv.cartToPolar(flow[...,0] , flow[...,1])
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv.normalize(mag , None , 0,255 , cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv , cv.COLOR_HSV2BGR)

        cv.imshow('RLOF optical flow' , bgr)
        cv.imshow('orginal' , frame)

        if cv.waitKey(10) & 0xFF == ord('q'): 
            break

        old_frame = frame

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    rlof_method('videos/Highway_6436.mp4')