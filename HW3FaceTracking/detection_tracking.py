import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices


######### Question 1 ##################
def CamShift_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    pt_x = c + np.int(w/2)
    pt_y = r + np.int(h/2)
    #cv2.rectangle(frame, (c, r), (c + w, r + h), [0, 0, 0], 2)
    #cv2.circle(frame, (pt_x, pt_y), 2, [0, 0, 0], -1)
    #cv2.imshow('image', frame)
    #cv2.waitKey(0)

    pt = 0, pt_x, pt_y
    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    #plt.plot(roi_hist)
    #plt.xlim([0,180])
    #plt.show()

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

	#*********************************************************
        #********************CAMshift ****************************
	#*********************************************************
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
	pt_x = np.int(ret[0][0])
	pt_y = np.int(ret[0][1])
	pt = frameCounter, pt_x, pt_y
	# to show box around face
        #pts = cv2.boxPoints(ret)
        #pts = np.int0(pts)
        ##print(pts)
        #img2 = cv2.polylines(frame,[pts],True, 255,2)
        #img2 = cv2.circle(frame,(pt_x, pt_y), 2, [0, 0, 0], -1)
        #cv2.imshow('image',img2)
        #plt.show()
        #k = cv2.waitKey(30)
 	######################################################


        # write the result to the output file
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

        cv2.destroyAllWindows()


    output.close()


######### Question 2 ##################
def Particle_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    im_h, im_w, im_c = frame.shape
    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    pt_x = c + np.int(w/2)
    pt_y = r + np.int(h/2)

    pt = 0, pt_x, pt_y
    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    #track_window = (c,r,w,h)

    # a function that, given a particle position, will return the particle's "fitness"
    def particleevaluator(back_proj, particle):
        return back_proj[particle[1],particle[0]]
    
    n_particles = 200
    
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos
    weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)
    #pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average
    #f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
    
    
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # --- tracking

        # Particle motion model: uniform step (TODO: find a better motion model)
        stepsize = 10
        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")
        
        # Clip out-of-bounds particles
        particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)
        
    	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    	hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        f = particleevaluator(hist_bp, particles.T) # Evaluate particles
        weights = np.float32(f.clip(1))             # Weight ~ histogram response
        weights /= np.sum(weights)                  # Normalize w
        pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average
	pt_x = pos[0] 
	pt_y = pos[1] 
	pt = frameCounter, pt_x, pt_y
	w_rect = np.ptp(particles[:,0])
	h_rect = np.ptp(particles[:,1])
        
        if 1. / np.sum(weights**2) < n_particles / 2. : # If particle cloud degenerate:
            particles = particles[resample(weights),:]  # Resample particles according to weights

	pt_x = pos[0] 
	pt_y = pos[1] 
	pt = frameCounter, pt_x, pt_y
	#img2 = cv2.rectangle(frame, (np.int0(pt_x - w_rect/2), np.int0(pt_y - h_rect/2)), (np.int0(pt_x + w_rect/2), np.int0(pt_y + h_rect/2)), 255, 2) 
	###for u in range(200):
        ###	img2 = cv2.circle(frame,(particles[u][0], particles[u][1]), 2, [0, 0, 255], -1)
        #img2 = cv2.circle(frame,(pt_x, pt_y), 2, [0, 0, 0], -1)
        #cv2.imshow('image',img2)
        ###plt.show()
        #k = cv2.waitKey(30)


        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

        cv2.destroyAllWindows()


    output.close()


######### Question 3 ##################
def Kalman_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    pt_x = c + np.int(w/2)
    pt_y = r + np.int(h/2)

    pt = 0, pt_x, pt_y
    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1
    track_window = (c,r,w,h)

    kalman = cv2.KalmanFilter(4,2,0)
    
    state = np.array([pt_x,pt_y,0,0], dtype='float64') # initial position
    current_measurement = np.array([pt_x,pt_y], dtype='float64') # initial position
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state



    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

	#print("pre = ",format(kalman.statePre))
	#print("post = ",format(kalman.statePost))
	#*********************************************************
        #********************Kalman ******************************
	#*********************************************************
	c,r,w,h = detect_one_face(frame)
    	pt_x = c + np.int(w/2)
    	pt_y = r + np.int(h/2)
	prediction = kalman.predict()
    	#state = np.array([pt_x,pt_y,0,0], dtype='float64')
	# --- tracking
	if c == 0 and r == 0 and w == 0 and h == 0:
		pt_x = prediction[0]
		pt_y = prediction[1]
		pt = frameCounter, pt_x, pt_y
	else:
		pt = frameCounter, pt_x, pt_y
		current_measurement[0] = pt_x
		current_measurement[1] = pt_y
		#prediction = kalman.predict()
		kalman.correct(current_measurement)

        # write the result to the output file
	#if frameCounter == 244:
        #img2 = cv2.circle(frame,(pt_x, pt_y), 2, [0, 0, 0], -1)
        #cv2.imshow('image',img2)
        ##plt.show()
        #k = cv2.waitKey(30)
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1


    output.close()


######### Question 4 ##################
def of_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100, qualityLevel = 0.01, minDistance = 10, blockSize = 7)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    frameCounter = 0
    # read first frame
    ret, old_frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(old_frame)

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

    # cv2.imshow("VideoFrame", old_frame)
    # plt.imshow(old_frame)
    # plt.show()
    # cv2.waitKey(30)

    # Write track point for first frame
    pt = frameCounter, np.int0(c + w / 2), np.int0(r + h / 2)
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        c, r, w, h = detect_one_face(frame)

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

            # write the result to the output file
            pt = frameCounter, np.int0(c + w / 2), np.int0(r + h / 2)
            output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        else:
            # Face not detected in the current frame.
            # print("== Face not detected in the %d frame ==" % frameCounter)
            # print(good_new)
            # Calculate the mean of all the co-ordinates. This should be the centre point of the face. To be done.
            OF_XY = np.mean(good_new, axis=0)
            pt = frameCounter, np.int0(OF_XY[0]), np.int0(OF_XY[1])
            output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
            cv2.circle(frame, (np.int0(OF_XY[0]), np.int0(OF_XY[1])), 3, (0, 255, 0), -1)

        #cv2.imshow("VideoFrame", frame)
        #k = cv2.waitKey(30) & 0xff
        #if k == 27:
            # break;

        frameCounter = frameCounter + 1

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        

    output.close()

if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        CamShift_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        Particle_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        Kalman_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        of_tracker(video, "output_of.txt")

'''
For Kalman Filter:

# --- init

state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state


# --- tracking

prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
'''

'''
For Particle Filter:

# --- init

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200

init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)


# --- tracking

# Particle motion model: uniform step (TODO: find a better motion model)
np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

# Clip out-of-bounds particles
particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

f = particleevaluator(hist_bp, particles.T) # Evaluate particles
weights = np.float32(f.clip(1))             # Weight ~ histogram response
weights /= np.sum(weights)                  # Normalize w
pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    particles = particles[resample(weights),:]  # Resample particles according to weights
# resample() function is provided for you
'''
