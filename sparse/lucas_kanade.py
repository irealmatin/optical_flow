import cv2
import numpy as np


def lucas_kanade_method(video_path):
    """
    Apply Lucas-Kanade Optical Flow method to track feature points in a video.
    this code find some feature points in the first frame using Shi-Tomasi(or Harris if shi-tomasi == False) corner detection method then 
    track those points in the next frames using Lucas-Kanade Optical Flow method. 
    """
    cap = cv2.VideoCapture(video_path)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(12, 12),
        maxLevel=2, # 1/4 
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), # cv2.TERM_CRITERIA_COUNT : stop when we reach limit iterate
    ) # cv2.TERM_CRITERIA_EPS : stop when change the result is very low . 

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params) # this function find the good feature to track using ShiTomasi method

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

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
            a, b = new.ravel() # or np.reshape(-1)
            c, d = old.ravel()

            a, b = int(a), int(b)
            c, d = int(c), int(d) # note : optical flow return the Coordinates of points in float type . but the drawing shape use integer . 

            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2) # draw line from old point to new point
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1) # draw circle at new point

        img = cv2.add(frame, mask)
        cv2.imshow("frame", img)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if k == ord("c"):
            mask = np.zeros_like(old_frame)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2) # our new points become the old points for next iteration



if __name__ == "__main__":
    lucas_kanade_method("videos/car.mp4")