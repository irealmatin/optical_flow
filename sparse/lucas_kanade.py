import cv2
import numpy as np


def lucaskanade_method(path):
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
    feature_params = dict(maxCorners=100, qualityLevel=0.4, minDistance=7, blockSize=7)

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
    ret, frame_init = cap.read()
    if not ret:
        print("Error: Video file not found or empty.")
        return

    old_gray = cv2.cvtColor(frame_init, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params) # this function find the good feature to track using ShiTomasi method

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame_init)
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
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()

            x_new, y_new = int(x_new), int(y_new)
            x_old, y_old = int(x_old), int(y_old) # note : optical flow return the Coordinates of points in float type . but the drawing shape use integer . 
            mask = cv2.line(mask, (x_new, y_new), (x_old, y_old), color[i].tolist(), 2) # draw line from old point to new point
            frame = cv2.circle(frame, (x_new, y_new), 5, color[i].tolist(), -1) # draw circle at new point

                        
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
        cv2.imshow("mask", mask)

        k = cv2.waitKey(60) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c"):
            mask = np.zeros_like(frame_init)
            
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2) # our new points become the old points for next iteration





def lucaskanade_manual_tracking(video_source):
    """
    Track a manually selected point using Lucas-Kanade Optical Flow.
    User selects a point with mouse click.
    """

    # --------------------------------------------------
    # 1. Video & Lucas-Kanade Parameters
    # --------------------------------------------------
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # --------------------------------------------------
    # 2. Read first frame
    # --------------------------------------------------
    ret, frame_init = cap.read()
    if not ret:
        print("Error: Empty video.")
        return

    frame_gray_init = cv2.cvtColor(frame_init, cv2.COLOR_BGR2GRAY)
    canvas = np.zeros_like(frame_init)

    # --------------------------------------------------
    # 3. State variables for tracking
    # --------------------------------------------------
    state = {
        "selected": False,
        "point": None,
        "old_points": None,
        "prev_gray": frame_gray_init,
    }

    # --------------------------------------------------
    # 4. Mouse callback
    # --------------------------------------------------
    def select_point(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["point"] = (x, y)
            state["selected"] = True
            state["old_points"] = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
            state["prev_gray"] = frame_gray_init.copy()
            #canvas[:] = 0  # Clear canvas on new selection

    cv2.namedWindow("Optical Flow")
    cv2.setMouseCallback("Optical Flow", select_point)

    # --------------------------------------------------
    # 5. Main Loop
    # --------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if state["selected"]:
            # Draw selected point
            cv2.circle(frame, state["point"], 5, (0, 0, 255), 2)

            # Lucas-Kanade
            new_points, st, _err = cv2.calcOpticalFlowPyrLK(
                state["prev_gray"],
                frame_gray,
                state["old_points"],
                None,
                **lk_params
            )

            if st[0] == 1: # if found the point
                x_new, y_new = new_points.ravel()
                x_old, y_old = state["old_points"].ravel()

                # Draw trajectory
                canvas[:] = cv2.line(
                    canvas,
                    (int(x_old), int(y_old)),
                    (int(x_new), int(y_new)),
                    (0, 255, 0),
                    2,
                )

                frame = cv2.circle(
                    frame,
                    (int(x_new), int(y_new)),
                    5,
                    (0, 255, 0),
                    -1,
                )

                # Update state
                state["old_points"] = new_points
                state["prev_gray"] = frame_gray.copy()
                state["point"] = (int(x_new), int(y_new))

        result = cv2.add(frame, canvas)
        cv2.imshow("Optical Flow", result)
        cv2.imshow("Canvas", canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            canvas[:] = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lucaskanade_method("sparse_optical_flow/videos/Highway_6436.mp4")  # 0 for webcam, or provide video file path
    # lucaskanade_manual_tracking("sparse_optical_flow/videos/Highway_6436.mp4")  # 0 for webcam, or provide video file path