# Instructions:
# Do not change the output file names, use the helper functions as you see fit

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Perspective warping")
   print("2 Cylindrical warping")
   print("3 Bonus perspective warping")
   print("4 Bonus cylindrical warping")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image1] " + "[path to input image2] " + "[path to input image3] " + "[output directory]")

'''
Detect, extract and match features between img1 and img2.
Using SIFT as the detector/extractor, but this is inconsequential to the user.

Returns: (pts1, pts2), where ptsN are points on image N.
    The lists are "aligned", i.e. point i in pts1 matches with point i in pts2.

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (pts1, pts2) = feature_matching(im1, im2)

    plt.subplot(121)
    plt.imshow(im1)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
    plt.subplot(122)
    plt.imshow(im2)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
'''
def feature_matching(img1, img2, savefig=False):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)

    matchesMask_ratio = [[0,0] for i in xrange(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in xrange(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]



    if savefig:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask_ratio_recip,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        plt.figure(),plt.xticks([]),plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])

'''
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching

Usage example:

    im = cv2.imread("myimage.jpg",0) #grayscale
    h,w = im.shape
    f = 700
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)
'''
def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0,0]

    im_h,im_w = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h,cyl_w = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0,cyl_w):
        for y_cyl in np.arange(0,cyl_h):
            theta = (x_cyl - x_c) / f
            h     = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K,X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
            cyl_mask[int(y_cyl),int(x_cyl)] = 255


    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png",bbox_inches='tight')
	plt.show()

    return (cyl,cyl_mask)

'''
Calculate the geometric transform (only affine or homography) between two images,
based on feature matching and alignment with a robust estimator (RANSAC).

Returns: (M, pts1, pts2, mask)
Where: M    is the 3x3 transform matrix
       pts1 are the matched feature points in image 1
       pts2 are the matched feature points in image 2
       mask is a binary mask over the lists of points that selects the transformation inliers

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (M, pts1, pts2, mask) = getTransform(im1, im2)

    # for example: transform im1 to im2's plane
    # first, make some room around im2
    im2 = cv2.copyMakeBorder(im2,200,200,500,500, cv2.BORDER_CONSTANT)
    # then transform im1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(im1, M, (im1.shape[1],im2.shape[0]), dst=im2.copy(), borderMode=cv2.BORDER_TRANSPARENT)

    plt.imshow(out, cmap='gray')
    plt.show()
'''
def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        #M = np.append(M, [[0,0,1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in xrange(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in xrange(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_

   
# ===================================================
# ================ Perspective Warping ==============
# ===================================================
def Perspective_warping(img1, img2, img3):
	
	# Write your codes here
	img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT, value=0)
	
	(M_2to1, pts1_2to1, pts2_2to1, mask_2to1) = getTransform(img2, img1, 'homography')
	(M_3to1, pts1_3to1, pts3_3to1, mask_3to1) = getTransform(img3, img1, 'homography')
 	img1 = cv2.warpPerspective(img2, M_2to1, (img1.shape[1],img1.shape[0]), dst=img1.copy(), borderMode=cv2.BORDER_TRANSPARENT)
	img1 = cv2.warpPerspective(img3, M_3to1, (img1.shape[1],img1.shape[0]), dst=img1.copy(), borderMode=cv2.BORDER_TRANSPARENT)
	#plt.imshow(img1, cmap='gray')
	#plt.show()

	#master = cv2.imread('/home/vvaid/CSE527/HW2/HW2-Panoramas/example_output1.png',0);
	#error = RMSD(img1, master)
	#print(error)

	output_name = sys.argv[5] + "output_homography.png"
	cv2.imwrite(output_name, img1)
	
	return True

	
def Bonus_perspective_warping(img1, img2, img3):
	
	# Write your codes here
	#output_image = img1 # This is dummy output, change it to your output
	A = img1.copy()
	B = img2.copy()
	C = img3.copy()
	A = cv2.copyMakeBorder(A,200,200,300,300, cv2.BORDER_CONSTANT, value=0)
	B = cv2.copyMakeBorder(B,200,200,0,600, cv2.BORDER_CONSTANT, value=0)
	C = cv2.copyMakeBorder(C,200,200,600,0, cv2.BORDER_CONSTANT, value=0)
	(M_BtoA, pts1_BtoA, pts2_BtoA, mask_BtoA) = getTransform(B, A, 'homography')
	(M_CtoA, pts1_CtoA, pt23_CtoA, mask_CtoA) = getTransform(C, A, 'homography')
	B_1 = cv2.warpPerspective(B, M_BtoA, (B.shape[1],B.shape[0]))
	C_1 = cv2.warpPerspective(C, M_CtoA, (C.shape[1],C.shape[0]))
	#plt.imshow(A, cmap='gray')
	#plt.show()
	#plt.imshow(B_1, cmap='gray')
	#plt.show()
	#plt.imshow(C_1, cmap='gray')
	#plt.show()
	m1 = np.zeros_like(A, dtype='float32')
    	m1[:,A.shape[1] * 4 / 10:] = 1 # make the mask half-and-half
	lpb1 = Laplacian_Pyramid_Blending_with_mask(A, C_1, m1, 4)
	#plt.imshow(lpb1, cmap='gray')
	#plt.show()

	m2 = np.zeros_like(A, dtype='float32')
    	m2[:,A.shape[1] * 6 / 10:] = 1 # make the mask half-and-half
	lpb2 = Laplacian_Pyramid_Blending_with_mask(B_1, lpb1, m2, 4)
	#plt.imshow(lpb2, cmap='gray')
	#plt.show()
	lpb2 = cv2.copyMakeBorder(lpb2, 0, 0,200,200, cv2.BORDER_CONSTANT, value=0)
	#print('lpb2 = {}', lpb2.shape)
	#plt.imshow(lpb2, cmap='gray')
	#plt.show()
	
	# Write out the result
	output_name = sys.argv[5] + "output_homography_lpb.png"
	cv2.imwrite(output_name, lpb2)
	
	return True


# ===================================================
# =============== Cynlindrical Warping ==============
# ===================================================
def Cylindrical_warping(img1, img2, img3):
	
	# Write your codes here

	#img1, img2 and img3 have the input images.

	#Do Cylindrical warping for the input images img1, img2 and img3.
	h,w = img1.shape
	f = 496
	K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
	(img1cyl, img1mask) = cylindricalWarpImage(img1, K)
	

	#Create the output_image variable that contains the output image with a padded border to accomodate the input images.	
	img1cyl = cv2.copyMakeBorder(img1cyl,50,50,300,300, cv2.BORDER_CONSTANT)
	img1mask =  cv2.copyMakeBorder(img1mask,50,50,300,300, cv2.BORDER_CONSTANT)
	#plt.imshow(img1cyl, cmap = "gray")
	#plt.show()

	h,w = img2.shape
	f = 496
	K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
	(img2cyl, img2mask) = cylindricalWarpImage(img2, K)

	h,w = img3.shape
	f = 496
	K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
	(img3cyl, img3mask) = cylindricalWarpImage(img3, K)

	(M_Img2cylToImg1cyl, pts1_Img2cylToImg1cyl, pts2_Img2cylToImg1cyl, mask_Img2cylToImg1cyl) = getTransform(img2cyl, img1cyl, "affine")
	(M_Img3cylToImg1cyl, pts1_Img3cylToImg1cyl, pts2_Img3cylToImg1cyl, mask_Img3cylToImg1cyl) = getTransform(img3cyl, img1cyl, "affine")

	warpimg2cyl = cv2.warpAffine(img2cyl, M_Img2cylToImg1cyl, (img1cyl.shape[1],img1cyl.shape[0]))
	warpimg2mask = cv2.warpAffine(img2mask, M_Img2cylToImg1cyl, (img1cyl.shape[1],img1cyl.shape[0]))
	warpimg2mask = cv2.bitwise_and(warpimg2mask, warpimg2mask, mask = cv2.bitwise_not(img1mask))

	#plt.imshow(warpimg2cyl, cmap = "gray")
	#plt.show()

	#plt.imshow(warpimg2mask, cmap = "gray")
        #plt.show()

	#plt.imshow(cv2.bitwise_and(warpimg2cyl, warpimg2cyl, mask = warpimg2mask), cmap = "gray")
        #plt.show()

	warpimg3cyl = cv2.warpAffine(img3cyl, M_Img3cylToImg1cyl, (img1cyl.shape[1],img1cyl.shape[0]))
	warpimg3mask = cv2.warpAffine(img3mask, M_Img3cylToImg1cyl, (img1cyl.shape[1],img1cyl.shape[0]))
	warpimg3mask = cv2.bitwise_and(warpimg3mask, warpimg3mask, mask = cv2.bitwise_not(img1mask))

	#plt.imshow(warpimg3cyl, cmap = "gray")
        #plt.show()

        #plt.imshow(warpimg3mask, cmap = "gray")
        #plt.show()

	#plt.imshow(cv2.bitwise_and(warpimg3cyl, warpimg3cyl, mask = warpimg3mask), cmap = "gray")
        #plt.show()

	img1cyl = cv2.bitwise_and(warpimg2cyl, warpimg2cyl, mask = warpimg2mask) + cv2.bitwise_and(warpimg3cyl, warpimg3cyl, mask = warpimg3mask) + img1cyl

	#plt.imshow(img1cyl, cmap = "gray")
	#plt.show()

	#master = cv2.imread("/home/snarasimha/HW2Panoramas/HW2-Panoramas/example_output2.png", 0)
	#print(img1cyl.shape)
	#print(master.shape)
	#error = RMSD(2, img1cyl, master)
	#print(error)

	# Write out the result
	output_image = img1cyl
	output_name = sys.argv[5] + "output_cylindrical.png"
	cv2.imwrite(output_name, output_image)
	
	return True

def Bonus_cylindrical_warping(img1, img2, img3):
	
	# Write your codes here
	#output_image = img1 # This is dummy output, change it to your output
	
	h,w = img1.shape
        f = 496
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
        (img1cyl, img1mask) = cylindricalWarpImage(img1, K)


        #Create the output_image variable that contains the output image with a padded border to accomodate the input images.
        img1cyl = cv2.copyMakeBorder(img1cyl,50,50,300,300, cv2.BORDER_CONSTANT)
        img1mask =  cv2.copyMakeBorder(img1mask,50,50,300,300, cv2.BORDER_CONSTANT)
        #plt.imshow(img1cyl, cmap = "gray")
        #plt.show()

        h,w = img2.shape
        f = 496
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
        (img2cyl, img2mask) = cylindricalWarpImage(img2, K)

        h,w = img3.shape
        f = 496
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
        (img3cyl, img3mask) = cylindricalWarpImage(img3, K)

        (M_Img2cylToImg1cyl, pts1_Img2cylToImg1cyl, pts2_Img2cylToImg1cyl, mask_Img2cylToImg1cyl) = getTransform(img2cyl, img1cyl, "affine")
        (M_Img3cylToImg1cyl, pts1_Img3cylToImg1cyl, pts2_Img3cylToImg1cyl, mask_Img3cylToImg1cyl) = getTransform(img3cyl, img1cyl, "affine")

        warpimg2cyl = cv2.warpAffine(img2cyl, M_Img2cylToImg1cyl, (img1cyl.shape[1],img1cyl.shape[0]))
        warpimg2mask = cv2.warpAffine(img2mask, M_Img2cylToImg1cyl, (img1cyl.shape[1],img1cyl.shape[0]))
        warpimg2mask = cv2.bitwise_and(warpimg2mask, warpimg2mask, mask = cv2.bitwise_not(img1mask))

	warpimg3cyl = cv2.warpAffine(img3cyl, M_Img3cylToImg1cyl, (img1cyl.shape[1],img1cyl.shape[0]))
        warpimg3mask = cv2.warpAffine(img3mask, M_Img3cylToImg1cyl, (img1cyl.shape[1],img1cyl.shape[0]))
        warpimg3mask = cv2.bitwise_and(warpimg3mask, warpimg3mask, mask = cv2.bitwise_not(img1mask))

	m1 = np.zeros_like(img1cyl, dtype='float32')
        m1[:, img1cyl.shape[1] * 4 / 10:] = 1 # make the mask half-and-half
        lpb1 = Laplacian_Pyramid_Blending_with_mask(img1cyl, warpimg3cyl, m1, 3)


	m2 = np.zeros_like(img1cyl, dtype='float32')
        m2[:,img1cyl.shape[1] * 6 / 10:] = 1 # make the mask half-and-half
        lpb2 = Laplacian_Pyramid_Blending_with_mask(warpimg2cyl, lpb1, m2, 3)
        #lpb2 = cv2.copyMakeBorder(lpb2,200,200,200,200, cv2.BORDER_CONSTANT, value=0)
        #print('lpb2 = {}', lpb2.shape)
        #plt.imshow(lpb2, cmap='gray')
        #plt.show()


	# Write out the result
	output_name = sys.argv[5] + "output_cylindrical_lpb.png"
	cv2.imwrite(output_name, lpb2)
	
	return True
	
'''
This exact function will be used to evaluate your results for HW2
Compare your result with master image and get the difference, the grading
criteria is posted on Piazza
'''
def RMSD(questionID, target, master):
    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        return -1
    else:
        nonZero_target = cv2.countNonZero(target)
        nonZero_master = cv2.countNonZero(master)

        if (questionID == 1):
           if (nonZero_target < 1200000):
               return -1
        elif(questionID == 2):
            if (nonZero_target < 700000):
                return -1
        else:
            return -1

        total_diff = 0.0;
        master_channels = cv2.split(master);
        target_channels = cv2.split(target);

        for i in range(0, len(master_channels), 1):
            dst = cv2.absdiff(master_channels[i], target_channels[i])
            dst = cv2.pow(dst, 2)
            mean = cv2.mean(dst)
            total_diff = total_diff + mean[0]**(1/2.0)

        return total_diff;

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) != 6):
      help_message()
      sys.exit()
   else: 
      question_number = int(sys.argv[1])
      if (question_number > 4 or question_number < 1):
	 print("Input parameters out of bound ...")
         sys.exit()
		 
   input_image1 = cv2.imread(sys.argv[2], 0)
   input_image2 = cv2.imread(sys.argv[3], 0)
   input_image3 = cv2.imread(sys.argv[4], 0) 

   function_launch = {
   1 : Perspective_warping,
   2 : Cylindrical_warping,
   3 : Bonus_perspective_warping,
   4 : Bonus_cylindrical_warping,
   }

   # Call the function
   function_launch[question_number](input_image1, input_image2, input_image3)
