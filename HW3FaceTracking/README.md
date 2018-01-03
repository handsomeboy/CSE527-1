# HW3: Detection and Tracking

Your goal is to:
* Detect the face in the first frame of the movie using pre-trained Viola-Jones detector
* Track the face throughout the movie using:
   1. CAMShift
   2. Particle Filter
   3. Face detector + Kalman Filter (always run the `kf.predict()`, and run `kf.correct()` when you get a new face detection)
Bonus (20pt): Face Detector + Optical Flow tracker (use the OF tracker whenever the face detector fails).

Due: Thu 10/19 9am

Skeleton and bootstrap code is provided.
HW3DetectionTracking.zip

We provide skeleton code to help detect faces, and build your tracker.
The code is bundled in the zip, see the file named detection_tracking.py.

To detect a face we have a convenience function:
```python
x,y,w,h = detect_one_face(frame) # rectangle of the face, or (0,0,0,0) if no face found
```

With the tracker skeleton you can basically just plug in the right tracker to get your functionality.
This is a skeleton of a tracker:
```python
def skeleton_tracker():
    # read video file
    v = cv2.VideoCapture("input.avi")
    
    output = open("output.txt","w")

    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos
    # see further explanation below
    
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        
        # perform the tracking
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()

        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        # write the result to the output file
        output.write("%d,%d\n" % pt)

    output.close()
```

OpenCV functions/objetcs to look into:
```
cv2.CamShift
cv2.KalmanFilter
cv2.calcOpticalFlowPyrLK
cv2.calcBackProject
```

## Kalman Filter

To save you the trouble of figuring it out, here's how to initialize the KF
```
state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman - cv2.KalmanFilter(4,2,0) # 4 state/hidden, 2 measurement, 0 control
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],  # a rudimentary constant speed model:
                                    [0., 1., 0., .1],  # x_t+1 = x_t + v_t
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)      # you can tweak these to make the tracker
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)      # respond faster to change and be less smooth
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state
```

During your KF tracking you will need to do something like the following:
```
prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
```

## Particle Filter
Here's how I initialize the PF with 200 particles:
```
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
```

During the PF tracking you will need to do something along the lines of
```
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
```

## Submission details

For each frame write out the tracking results to a text file, in the format specified below.
```
0,x_0,y_0
1,x_1,y_1
2,x_2,y_2
...
256,x_n,y_n
```
Plain text file with frame_index and x,y coordinates of the middle of the face/head for each frame in the movie (no space between).
Note: you also need to write the initial first position, i.e. the first frame face detection (a total of 257 frames). Make sure you have the correct number of frames in your .txt file.

Write 3 (or 4) outputs for each condition:
* output_camshift.txt
* output_particle.txt
* output_kalman.txt
* Bonus: output_of.txt

Put your results in a zip with your id as the filename.
Results are checked automatically, so pay extra attention to file naming conventions as well as formats.

An example folder structure for HW3 is uploaded and available under "Resources": Bilbo_Baggins_111123456.zip
It is recommended to download that file, replace the original files with your files, change the name and submit it.

The grades for HW3 will be calculated based on the difference between your tracking results and ground truth results.
We will release the ground truth and the function computing your grades after the deadline of HW3. A visually correct tracking result should be sufficient for a good mark.

Please keep the interface of the program intact and we will test your program under VM (make sure to get rid of all unnecessary dependencies). If the generated results does not match the one in "Result" folder, you will receive 0 mark for that question. It is critical to have the correct number of frames in your results.
