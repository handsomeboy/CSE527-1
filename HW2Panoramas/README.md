# HW2: Image Alignment, Panoramas

Your goal is to create 2 panoramas:
1. Using homographies and perspective warping on a common plane (3 images).
2. Using cylindrical warping (many images).

In both options you should:
* Read in the images: input1.jpg, input2.jpg, input3.jpg
* [Apply cylindrical wrapping if needed]
* Calculate the transformation (homography for projective; affine for cylindrical) between each
* Transform input2 and input3 to the plane of input1, and produce output.png
* Bonus (!!): Use your Laplacian Blending code to stitch the images together nicely

Use the supplied code to work out through this assignment.
You have functions to get a cylindrical wrapping as well as calculate the geometric transform.

## Submission guidelines:
Here are skeleton code, helper code and input images:
HW2Panoramas.zip

Use the provided code and usage examples to help you.
You final code for each case should be around 15-20 lines of python code (that's what my code is). In C++ you will need to add more LoC, but not much more.
Use the solutions key (watermarked) to see if you're in the right direction.

You'll find the following functions useful:
```python
cv2.warpPerspective()
cv2.warpAffine()
 ```
Inputs are numbered: `input1.jpg`, `input2.jpg`, `input3.jpg`
Outputs should be:
### output_homography.png - for the projective stitch
* Output image size should be: 1608 × 1312
* Use `cv2.copyMakeBorder()` to add this padding to allow for space around the center image to paste the other (transformed) images:
* `out = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)`

### output_cylindrical.png - for the cylindrical stitch
* Output image size should be: 1208 × 1012
* `out = cv2.copyMakeBorder(img1cyl,50,50,300,300, cv2.BORDER_CONSTANT)`

If you are going after the LPB for blending, make a separate set of outputs:
* output_homography_lpb.png
* output_cylindrical_lpb.png
* [Here](http://www.morethantechnical.com/2017/09/29/laplacian-pyramid-with-masks-in-opencv-python/) is (my) code to do LBP with masks:

Make sure your output filenames match this convention.

## Grading criteria for HW2
1. You should use the skeleton codes and do the implementation on it.
2. Do not change the output file names in skeleton codes.
3. An example of folder structure is uploaded under resources.

* Not complying to the folder structure to any extent will lead to 5 points off.
* A RMSD (Root of the Mean Squared Difference) function is supplied to calculate how different your results are from master images:
       - 0 - 20: full credits, 20 included
       - 20 - 35: (1 - (your score - 20) / 20) * full credits, 35 included
       - beyond 35: 0
* All questions regarding grading will not be answered.
* The same RMSD function will also be used to calculate the difference between the images in your "Results" folder and the outputs from running your program. If your program fails to run or produces images that do not match the ones in your folder, you will get 0 points.

The bonus question is optional, the award for bonus question is still under discussion. Make sure you also have correct outputs for standard methods even if you decide to do bonus questions.

Due: Thu 10/5 9:00 am. Submit on blackboard.
