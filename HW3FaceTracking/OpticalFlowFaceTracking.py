import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def FaceDetector(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0, 0, 0, 0)
    return faces[0]


def OpticalFlowTracker(v):

    # Parameters for ShiTomasi Corner Detection
    feature_params = dict(maxCorners = 100, qualityLevel = 0.01, minDistance = 10, blockSize = 7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    frameCounter = 0
    # read first frame
    ret, old_frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c, r, w, h = FaceDetector(old_frame)

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    maskForFeatureTracking = np.zeros_like(old_gray)
    maskForFeatureTracking[r+15:r+h-10, c+20:c+w-20] = 255
    # cv2.imshow("MaskedImage", maskForFeatureTracking)
    # plt.imshow(maskForFeatureTracking)
    # plt.show()

    # The below line is working, however the matched features start from (0, 0) rather than from the window. To be analyzed yet.
    # p0 = cv2.goodFeaturesToTrack(old_gray[c : c + w, r : r + h], mask = None, **feature_params)
   
    # print(old_gray.shape)
    # print(maskForFeatureTracking.shape) 
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = maskForFeatureTracking, **feature_params)

    p0int = np.int0(p0)

    for i in p0int:
        x, y = i.ravel()
        cv2.circle(old_frame, (x, y), 3, 255, -1)

    cv2.rectangle(old_frame, (c, r), (c + w, r + h), (0, 0, 255), 2)

    # print(p0)

    cv2.imshow("VideoFrame", old_frame)
    # plt.imshow(old_frame)
    # plt.show()
    cv2.waitKey(30)

    # Write track point for first frame
    pt = frameCounter, np.int0(c + w / 2), np.int0(r + h / 2)
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c, r, w, h)

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        c, r, w, h = FaceDetector(frame)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        good_new_int = np.int0(good_new)
        for i in good_new_int:
            x, y = i.ravel()
            cv2.circle(frame, (x, y), 3, 255, -1)

        # if you track a rect (e.g. face detector) take the mid point,
	if c != 0 and r != 0 and w != 0 and h != 0:
            # Face detected in the current frame

            # Draw the rectangle to show face is detected.
            cv2.rectangle(frame, (c, r), (c + w, r + h), (0, 0, 255), 2)
            cv2.circle(frame, (np.int0(c + w / 2), np.int0(r + h / 2)), 3, (0, 0, 255), -1) 

        else:
            # Face not detected in the current frame.
            # print("== Face not detected in the %d frame ==" % frameCounter)
            # print(good_new)
            # Calculate the mean of all the co-ordinates. This should be the centre point of the face. To be done.
            OF_XY = np.mean(good_new, axis=0)
            pt = frameCounter, np.int0(OF_XY[0]), np.int0(OF_XY[1])
            cv2.circle(frame, (np.int0(OF_XY[0]), np.int0(OF_XY[1])), 3, (0, 255, 0), -1)

        cv2.imshow("VideoFrame", frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
             break;

        frameCounter = frameCounter + 1

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        


if __name__ == '__main__':
   
    # Check for the correctness of the input arguments.
    if (len(sys.argv) != 2):
        print("Usage: OpticalFlowFaceTracking.py [Input_Video]")
        print("    -> [Input_Video] : Relative path to the input video")
        print("Example usage: " + sys.argv[0] + "input.avi")
        sys.exit()

    # Read the input video into the variable video.
    video = cv2.VideoCapture(sys.argv[1]);

    # Track the face using face tracker and Optical Flow techniques.
    OpticalFlowTracker(video)

