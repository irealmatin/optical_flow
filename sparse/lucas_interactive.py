import numpy as np
import cv2
import time

def optimized_lucas_kanade(source=0):
    """
    Runs Lucas-Kanade Optical Flow with Forward-Backward Error Checking 
    and Trajectory History.
    """
    
    # 1. Configuration Parameters
    # ---------------------------
    # Params for corner detection
    feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    # Params for Lucas-Kanade flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    
    trajectory_len = 40    # Keep last 40 points (Snake length)
    detect_interval = 5    # Find new points every 5 frames
    trajectories = []      # List to store track history
    frame_idx = 0          # this is for counting frames
    
    # 2. Initialization
    # -----------------
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        start_time = time.time()
        
        # Read Frame
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis_frame = frame.copy() # Visualization copy : this is for drawing purpose

        # 3. Optical Flow Tracking (if we have points)
        # --------------------------------------------
        if len(trajectories) > 0:
            img0, img1 = prev_gray, frame_gray
            
            # Get the last point of each trajectory to track
            p0 = np.float32([trj[-1] for trj in trajectories]).reshape(-1, 1, 2)
            
            # A. Forward Flow: Frame (t-1) -> Frame (t)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            
            # B. Backward Flow: Frame (t) -> Frame (t-1) (Validation Check)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            
            # C. Check Validity: Distance between original and back-tracked point
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            is_good_point = d < 1  # Valid if error is less than 1 pixel

            new_trajectories = [] # Temporary list for updated trajectories
            
            # D. Update Trajectories
            for trj, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), is_good_point):
                if not good_flag:
                    continue # Drop bad points
                
                trj.append((x, y)) # Add new point to trajectory
                
                # Trim the tail if it's too long
                if len(trj) > trajectory_len:
                    del trj[0] # remove the oldest point
                
                new_trajectories.append(trj)
                
                # Draw the head of the snake
                cv2.circle(vis_frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            trajectories = new_trajectories # Update the main trajectory list

            # Draw the tail (Polylines)
            cv2.polylines(vis_frame, [np.int32(trj) for trj in trajectories], False, (0, 255, 0))
            
            # Show count
            cv2.putText(vis_frame, f'Track count: {len(trajectories)}', (20, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 4. Detect New Points (Replenishment)
        # ------------------------------------
        if frame_idx % detect_interval == 0:
            # Create a mask to avoid detecting points on existing tracks
            mask = np.zeros_like(frame_gray)
            mask[:] = 255 # Fill with white
            
            # Draw black circles at current tracking positions
            for trj in trajectories:
                x, y = trj[-1]
                cv2.circle(mask, (int(x), int(y)), 5, 0, -1)

            # Detect new features in the white areas
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    trajectories.append([(x, y)]) # Start a new trajectory

        # 5. Housekeeping
        # ---------------
        frame_idx += 1
        prev_gray = frame_gray
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Advanced Optical Flow', vis_frame)
        cv2.imshow('Mask', mask)

         # Exit on 'q' key
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    optimized_lucas_kanade()