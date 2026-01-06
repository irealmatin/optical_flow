import cv2
import numpy as np


def lucas_kanade_method(path):
    """
    Apply Lucas-Kanade Optical Flow method to track feature points in a video.
    this code find some feature points in the first frame using Shi-Tomasi(or Harris if shi-tomasi == False) corner detection method then 
    track those points in the next frames using Lucas-Kanade Optical Flow method. 
    """
    #-------------------------------------------------------
    # 1. Video Capture and Parameters
    #-------------------------------------------------------
    cap = cv2.VideoCapture(path)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2, # 1/4 
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), # cv2.TERM_CRITERIA_COUNT : stop when we reach limit iterate
    ) # cv2.TERM_CRITERIA_EPS : stop when change the result is very low . 

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    #-------------------------------------------------------
    # 2. Take first frame and find corners in it
    #-------------------------------------------------------
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Video file not found or empty.")
        return
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params) # this function find the good feature to track using ShiTomasi method

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    #-------------------------------------------------------
    # 3. Optical Flow Calculation Loop
    #-------------------------------------------------------
    while True: 
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        ) # p1 : new point positions, st : status array ( if found the point or not ) , err : error value

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
           
            a, b = int(a), int(b)
            c, d = int(c), int(d) # note : optical flow return the Coordinates of points in float type . but the drawing shape use integer . 

            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2) # draw line from old point to new point
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1) # draw circle at new point

                        
            #-----------------------------------------------------------
            
            ## Using shift to improve sub-pixel accuracy in drawing
            ## Convert float coordinates to fixed-point representation
            # shift = 4  
            # a_f, b_f = new.ravel()
            # c_f, d_f = old.ravel()

            # a = int(a_f * (1 << shift))
            # b = int(b_f * (1 << shift))
            # c = int(c_f * (1 << shift))
            # d = int(d_f * (1 << shift))

            # mask = cv2.line(
            #     mask,
            #     (a, b),
            #     (c, d),
            #     color[i].tolist(),
            #     2,
            #     shift=shift,
            #     lineType=cv2.LINE_AA
            # )

            # frame = cv2.circle(
            #     frame,
            #     (a, b),
            #     5,
            #     color[i].tolist(),
            #     -1,
            #     shift=shift,
            #     lineType=cv2.LINE_AA
            # )

            # -----------------------------------------------------------
            
        img = cv2.add(frame, mask) # this is for overlaying the line on the frame
        cv2.imshow("frame", img)

        k = cv2.waitKey(60) & 0xFF
        if k == 27:
            break
        if k == ord("c"):
            mask = np.zeros_like(old_frame)
            
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2) # our new points become the old points for next iteration



if __name__ == "__main__":
    lucas_kanade_method("videos/car.mp4")  # 0 for webcam, or provide video file path