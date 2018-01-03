# HW5: Structured Light
Finally! It is here: Homework assignment 5.
I hope it will live up to your expectations!

Your goal is to reconstruct a scene from multiple structured light scannings of it.
* Calibrate projector with the “easy” method
   1. Use ray-plane intersection
   2. Get 2D-3D correspondence and use stereo calibration
   3. We did this work for you...
* Get the binary code for each pixel - this you should do, but it's super easy
* Correlate code with (x,y) position - we provide a "codebook" from binary code -> (x,y)
* With 2D-2D correspondence
   1. Perform stereo triangulation (existing function) to get a depth map
   2. You do this too
 

HW5StructredLight.zip

Due: Tuesday 11/21 9am.

 

## Update Nov 19

Bonus Points - 10 points
* Add color to your 3D cloud
* When finding correspondences, take the RGB values from "aligned001.png"
* Add them later to your reconstruction
* Output a file called "output_color.xyzrgb" with the following format
   1. "%d %d %d %d %d %d\n"%(x, y, z, r, g, b)
   2. for each 3D+RGB point
 
## Update Nov 16 2017
Note: We have compiled a new dataset of images - it is much more aligned and results in a lot less outliers.
New image dataset: images_new.zip
New reference reconstruction: `sample_output_new.xyz`
Also, if you're interested in how we did the projector-camera calibration, I've written a blog post about it [here](http://www.morethantechnical.com/2017/11/17/projector-camera-calibration-the-easy-way/)

You will be given:
* Stereo calibration
   1. K and distortion for projector and camera
   2. R, and t between projector and camera
* Images of binary codes pattern
   1. For depth scanning
* Skeleton code
   1. It has "TODO"s that guide you through (part of) the process of S-L stereo reconstruction
* Example output to match with

There are going to be a lot of errors decoding the patters.

Add this filtering after you the x_projector and y_projector within the  loop:
```python
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code
            # ...
            # ... obtain x_p, y_p - the projector coordinates from the codebook
            if x_p >= 1279 or y_p >= 799: # filter
                continue

# ...
# after cv2.triangulatePoints and cv2.convertPointsFromHomogeneous
# apply another filter on the Z-component
mask = (points_3d[:,:,2] > 200) & (points_3d[:,:,2] < 1400)
```

## Submission Details
Submit your codes and output.xyz file in a zip to blackboard.
As always make sure you maintain the proper filename for submission.
The homework can be completed with another 20-30 lines of codes given the skeleton codes.

Folder structure:
Failing to follow the required folder structure will lead to 5 points off. An example folder created specifically for HW5 can be found under "Resources" which is named as "Aragorn_Elessar_111134567.zip"

Submission requirements:
1. Keep the source file name unchanged as "reconstruct.py"
2. Do not submit the images or any other file already given to you
3. ** It is assumed that the folder "images" (containing source images), binary_codes_ids_codebook.pckl, stereo_calibration.pckl are in the same folder as your source file "reconstruct.py" when we test your program
4. Do NOT change main function or `write_3d_points()`

## Update Nov 14 2017
The write function for 3d points should be 
```
f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))
```
The sample output.xyz produced with the previous write function is not correct.

## Update Nov 19 2017 IMPORTANT
Please also submit the correspondence image under "Results" folder (name it as correspondence.jpg). The correspondence image and output.xyz will each carry half points for HW5. It is highly recommended to visualize your output.xyz (with meshLab or CloudCompare). The correct sample output.xyz will shortly be uploaded. Please make sure to filter the outlier points.

## Update Nov 20 2017
The provided output.xyz is for you to get a rough idea how the result should look like. It is not there to be compared. We will make a new one when grading and release it for your reference later.
