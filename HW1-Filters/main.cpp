// Instructions:
// For question 1, only modify function: histogram_equalization
// For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
// For question 3, only modify function: laplacian_pyramid_blending

#include "./header.h"

using namespace std;
using namespace cv;

void help_message(char* argv[])
{
   cout << "Usage: [Question_Number] [Input_Options] [Output_Options]" << endl;
   cout << "[Question Number]" << endl;
   cout << "1 Histogram equalization" << endl;
   cout << "2 Frequency domain filtering" << endl;
   cout << "3 Laplacian pyramid blending" << endl;
   cout << "[Input_Options]" << endl;
   cout << "Path to the input images" << endl;
   cout << "[Output_Options]" << endl;
   cout << "Output directory" << endl;
   cout << "Example usages:" << endl;
   cout << argv[0] << " 1 " << "[path to input image] " << "[output directory]" << endl;
   cout << argv[0] << " 2 " << "[path to input image1] " << "[path to input image2] " << "[output directory]" << endl;
   cout << argv[0] << " 3 " << "[path to input image1] " << "[path to input image2] " << "[output directory]" << endl;
}

// ===================================================
// ======== Question 1: Histogram equalization =======
// ===================================================

Mat histogram_equalization(const Mat& img_in)
{
   // Write histogram equalization codes here
   Mat img_out;
   Mat Image = img_in;
   Mat Image3Channels[3];
   Mat equalizedImageBlue = Mat::zeros(Image.rows, Image.cols, CV_32FC1);
   Mat equalizedImageGreen = Mat::zeros(Image.rows, Image.cols, CV_32FC1);
   Mat equalizedImageRed = Mat::zeros(Image.rows, Image.cols, CV_32FC1);
   split(Image,Image3Channels);
   
   //--Histogram calculation--
   Mat histBlue = Mat::zeros(256, 1, CV_32FC1);
   Mat histGreen = Mat::zeros(256, 1, CV_32FC1);
   Mat histRed = Mat::zeros(256, 1, CV_32FC1);
   Mat cdfBlue = Mat::zeros(256, 1, CV_32FC1);	
   Mat cdfGreen = Mat::zeros(256, 1, CV_32FC1);	
   Mat cdfRed = Mat::zeros(256, 1, CV_32FC1);	
   int histSizes[] = {256};
   float range[] = {0, 255};
   const float* ranges[] = {range}; 
   
   calcHist(&Image3Channels[0], 1, 0, Mat(), histBlue, 1, histSizes, ranges, true, false);	
   calcHist(&Image3Channels[1], 1, 0, Mat(), histGreen, 1, histSizes, ranges, true, false);	
   calcHist(&Image3Channels[2], 1, 0, Mat(), histRed, 1, histSizes, ranges, true, false);	
   
   normalize(histBlue, histBlue, 1.0, 0.0, NORM_L1);
   normalize(histGreen, histGreen, 1.0, 0.0, NORM_L1);
   normalize(histRed, histRed, 1.0, 0.0, NORM_L1);
   
   //--Calculating cdf from pdf--
   for(int i = 0; i <= 255; i++) {
   	for( int j = 0; j <= i; j++){
   		cdfBlue.at<float>(i, 0) = cdfBlue.at<float>(i, 0) + histBlue.at<float>(j, 0);  	
   		cdfGreen.at<float>(i, 0) = cdfGreen.at<float>(i, 0) + histGreen.at<float>(j, 0);  	
   		cdfRed.at<float>(i, 0) = cdfRed.at<float>(i, 0) + histRed.at<float>(j, 0);  	
   	}	
   }
   
   //--Getting new equalized pixel values from cdf--
   for(int i = 0; i < Image.rows; i++){
   	for(int j = 0; j < Image.cols; j++){
   	equalizedImageBlue.at<float>(i, j) = cdfBlue.at<float>(Image3Channels[0].at<uchar>(i,j), 0); 
   	equalizedImageGreen.at<float>(i, j) = cdfGreen.at<float>(Image3Channels[1].at<uchar>(i,j), 0); 
   	equalizedImageRed.at<float>(i, j) = cdfRed.at<float>(Image3Channels[2].at<uchar>(i,j), 0); 
   	}
   }

   //--Merging the 3 channels-- 
   Mat FinalColorImage1;
   vector<Mat> eqImage1;
   eqImage1.push_back(equalizedImageBlue);
   eqImage1.push_back(equalizedImageGreen);
   eqImage1.push_back(equalizedImageRed);
   merge(eqImage1, FinalColorImage1);
   convertScaleAbs(FinalColorImage1, img_out, 255.0);
   
   return img_out;


}

bool Question1(char* argv[])
{
   // Read in input images
   Mat input_image = imread(argv[2], IMREAD_COLOR);

   // Histogram equalization
   Mat output_image = histogram_equalization(input_image);

   // Write out the result
   string output_name = string(argv[3]) + string("output1.jpg");
   imwrite(output_name.c_str(), output_image);

   return true;
}

// ===================================================
// ===== Question 2: Frequency domain filtering ======
// ===================================================


void shiftDFT(Mat &planes)
{
   planes = planes(Rect(0, 0, planes.cols & -2, planes.rows & -2));
   int height = planes.rows / 2;
   int width = planes.cols / 2;

   Mat Q0 = planes(Rect( 0, 0, width, height));
   Mat Q1 = planes(Rect(width, 0, width, height));
   Mat Q2 = planes(Rect(0, height, width, height));
   Mat Q3 = planes(Rect(width, height, width, height));
   
   Mat temp;
   Q0.copyTo(temp);
   Q3.copyTo(Q0);
   temp.copyTo(Q3);
   
   Q1.copyTo(temp);
   Q2.copyTo(Q1);
   temp.copyTo(Q2);
}

Mat low_pass_filter(const Mat& img_in)
{
   // Write low pass filter codes here

   Mat img_out;
   //--Finding DFT--
   Mat padded = Mat::zeros(img_in.rows, img_in.cols, CV_32FC1);
   int newRows = getOptimalDFTSize(img_in.rows);
   int newCols = getOptimalDFTSize(img_in.cols);
   copyMakeBorder(img_in, padded, 0, newRows - img_in.rows, 0, newCols - img_in.cols, BORDER_CONSTANT, Scalar::all(0));
   
   Mat planes[2] = {Mat_<float>(padded), Mat::zeros(newRows, newCols, CV_32FC1)};
   Mat complexImage;
   merge(planes, 2, complexImage);
   
   dft(complexImage, complexImage);
   split(complexImage, planes);
   
   //--Shift DFT--
   shiftDFT(planes[0]);
   shiftDFT(planes[1]);
   
   //--Creating mask--
   Mat Mask(newRows, newCols, CV_32FC1, Scalar(0));
   int MaskSize = 20;
   Mask(Rect(complexImage.cols/2 - (MaskSize/2), complexImage.rows/2 - (MaskSize/2), MaskSize, MaskSize)) = 1;	
   
   //--Applying Mask--
   planes[0] = planes[0].mul(Mask);
   planes[1] = planes[1].mul(Mask);
   
   shiftDFT(planes[0]);
   shiftDFT(planes[1]);
   
   Mat AfterFFT;
   merge(planes, 2, AfterFFT);
   
   //--IFFT--
   Mat inverseTransform;
   dft(AfterFFT, inverseTransform, DFT_INVERSE|DFT_REAL_OUTPUT);
   normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
   convertScaleAbs(inverseTransform, img_out, 255.0);

   return img_out;

}

Mat high_pass_filter(const Mat& img_in)
{
   // Write high pass filter codes here

   Mat img_out;
   //--Finding DFT--
   Mat padded = Mat::zeros(img_in.rows, img_in.cols, CV_32FC1);
   int newRows = getOptimalDFTSize(img_in.rows);
   int newCols = getOptimalDFTSize(img_in.cols);
   copyMakeBorder(img_in, padded, 0, newRows - img_in.rows, 0, newCols - img_in.cols, BORDER_CONSTANT, Scalar::all(0));
   
   Mat planes[2] = {Mat_<float>(padded), Mat::zeros(newRows, newCols, CV_32FC1)};
   Mat complexImage;
   merge(planes, 2, complexImage);
   
   dft(complexImage, complexImage);
   split(complexImage, planes);
   
   //--Shift DFT to get low freq components at the image center--
   shiftDFT(planes[0]);
   shiftDFT(planes[1]);
   
   //--Creating mask--
   Mat Mask(newRows, newCols, CV_32FC1, Scalar(1));
   int MaskSize = 20;
   Mask(Rect(complexImage.cols/2 - (MaskSize/2), complexImage.rows/2 - (MaskSize/2), MaskSize, MaskSize)) = 0;	
   
   //--Apply Mask--
   planes[0] = planes[0].mul(Mask);
   planes[1] = planes[1].mul(Mask);
   
   shiftDFT(planes[0]);
   shiftDFT(planes[1]);
   
   Mat AfterFFT;
   merge(planes, 2, AfterFFT);
   
   //--IFFT--
   Mat inverseTransform;
   dft(AfterFFT, inverseTransform, DFT_INVERSE|DFT_REAL_OUTPUT);
   normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
   convertScaleAbs(inverseTransform, img_out, 255.0);

   return img_out;

}


Mat deconvolution(const Mat& img_in)
{
   // Write deconvolution codes here

   Mat img_out;   
   //namedWindow("OriginalImage", WINDOW_AUTOSIZE);
   //imshow("OriginalImage", img_in);
   //waitKey(0);

   Mat I = img_in.clone();

   Mat padded;
   int m = getOptimalDFTSize( I.rows );
   int n = getOptimalDFTSize( I.cols );
   copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

   Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
   Mat complexI;
   merge(planes, 2, complexI);

   dft(complexI, complexI);
   split(complexI, planes);

   //Get the Gaussian kernel here.
   Mat GaussianKernelOneD = getGaussianKernel(21, 5, CV_32FC1);
   normalize(GaussianKernelOneD, GaussianKernelOneD, 0, 1, CV_MINMAX);
   Mat GaussianKernelTwoD = GaussianKernelOneD * GaussianKernelOneD.t();
   Mat GaussianKernel = GaussianKernelTwoD;
   //imshow("GaussianKernel", GaussianKernel);
   //waitKey(0);
   //cout << "SizeOf(GaussianKernel) = " << GaussianKernel.size() << endl;

   //Get the FT of GaussianKernel.
   //Mat paddedGaussian;
   int o = getOptimalDFTSize(GaussianKernel.rows );
   int p = getOptimalDFTSize(GaussianKernel.cols );
   Mat paddedGaussian = Mat::zeros(m, n, CV_32FC1);
   //Mat blah(paddedGaussian, Rect(cvFloor((n - GaussianKernel.cols)/2), cvFloor((m - GaussianKernel.rows)/2), GaussianKernel.cols, GaussianKernel.rows));
   //GaussianKernel.copyTo(blah);
   copyMakeBorder(GaussianKernel, paddedGaussian, 0, m - GaussianKernel.rows, 0, n - GaussianKernel.cols, BORDER_CONSTANT, Scalar::all(0));
   Mat planesKernel[] = {Mat_<float>(paddedGaussian), Mat::zeros(paddedGaussian.size(), CV_32F)};
   Mat complexKernel;
   merge(planesKernel, 2, complexKernel);
   dft(complexKernel, complexKernel);

   Mat finalOutput;
   mulSpectrums(complexI, complexKernel, finalOutput, 0, CV_DXT_MUL_CONJ);
   //Calculate the IDFT of the DFT calculated above and display it.
   Mat inverseTransform;
   dft(finalOutput, inverseTransform, DFT_INVERSE | DFT_REAL_OUTPUT);
   normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
   convertScaleAbs(inverseTransform, img_out, 255.0);
   //imshow("Reconstructed", img_out);
   //waitKey();
   
   return img_out;

}

bool Question2(char* argv[])
{
   // Read in input images
   Mat input_image1 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
   Mat input_image2 = imread(argv[3], IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

   // Low and high pass filters
   Mat output_image1 = low_pass_filter(input_image1);
   Mat output_image2 = high_pass_filter(input_image1);

   // Deconvolution
   Mat output_image3 = deconvolution(input_image2);

   // Write out the result
   string output_name1 = string(argv[4]) + string("output2LPF.jpg");
   string output_name2 = string(argv[4]) + string("output2HPF.jpg");
   string output_name3 = string(argv[4]) + string("output2deconv.jpg");
   imwrite(output_name1.c_str(), output_image1);
   imwrite(output_name2.c_str(), output_image2);
   imwrite(output_name3.c_str(), output_image3);

   return true;
}

// ===================================================
// ===== Question 3: Laplacian pyramid blending ======
// ===================================================

Mat laplacian_pyramid_blending(const Mat& img_in1, const Mat& img_in2)
{
   // Write laplacian pyramid blending codes here
   Mat img_out; // Blending result

   //Copy both the LeftImage and RightImage to Mat.
   Mat LeftImage = img_in1;
   Mat RightImage = img_in2;

   //namedWindow("OriginalImages", WINDOW_AUTOSIZE);
   //imshow("OriginalImages", LeftImage);
   //waitKey(0);
   //namedWindow("OriginalImages", WINDOW_AUTOSIZE);
   //imshow("OriginalImages", RightImage);
   //waitKey(0);
   //destroyAllWindows();


   //Make sure that the two images are of the same size. Else make the 2 images of the same size.
   if (LeftImage.size() != RightImage.size()) {
      LeftImage = Mat(LeftImage, Range::all(), Range(0, LeftImage.rows));
      RightImage = Mat(RightImage, Range(0, LeftImage.rows), Range(0, LeftImage.rows));
   }

   //namedWindow("ImagesOfSameSize", WINDOW_AUTOSIZE);
   //imshow("ImagesOfSameSize", LeftImage);
   //waitKey(0);
   //namedWindow("ImagesOfSameSize", WINDOW_AUTOSIZE);
   //imshow("ImagesOfSameSize", RightImage);
   //waitKey(0);
   //destroyAllWindows();

   //Convert the pixels in the Mat to store float values.
   Mat_<Vec3f> LeftImageInFloat, RightImageInFloat;
   LeftImage.convertTo(LeftImageInFloat, CV_32F, 1.0 / 255.0);
   RightImage.convertTo(RightImageInFloat, CV_32F, 1.0 / 255.0);

   //namedWindow("LeftGaussianPyramid", WINDOW_AUTOSIZE);
   //Create Gaussian Pyramid for the left image.
   vector<Mat_<Vec3f> > LeftGaussianPyramid;
   LeftGaussianPyramid.push_back(LeftImageInFloat);
   //imshow("LeftGaussianPyramid", LeftGaussianPyramid.back());
   //waitKey(0);
   Mat CurrentImage = LeftImageInFloat;
   for (int i = 0; i < 6; i++) {
      Mat OneLevelDown;
      pyrDown(CurrentImage, OneLevelDown);
      LeftGaussianPyramid.push_back(OneLevelDown);
      CurrentImage = OneLevelDown;
      //imshow("LeftGaussianPyramid", OneLevelDown);
      //waitKey(0);
   }
   //destroyAllWindows();

   //namedWindow("RightGaussianPyramid", WINDOW_AUTOSIZE);
   //Create Gaussian Pyramid for the left image.
   vector<Mat_<Vec3f> > RightGaussianPyramid;
   RightGaussianPyramid.push_back(RightImageInFloat);
   //imshow("RightGaussianPyramid", RightGaussianPyramid.back());
   //waitKey(0);
   CurrentImage = RightImageInFloat;
   for (int i = 0; i < 6; i++) {
      Mat OneLevelDown;
      pyrDown(CurrentImage, OneLevelDown);
      RightGaussianPyramid.push_back(OneLevelDown);
      CurrentImage = OneLevelDown;
      //imshow("RightGaussianPyramid", OneLevelDown);
      //waitKey(0);
   }
   //destroyAllWindows();


   //namedWindow("LeftLaplacianPyramid", WINDOW_AUTOSIZE);
   //Create Laplacian Pyramid for the left image.
   vector<Mat_<Vec3f> > LeftLaplacianPyramid;
   LeftLaplacianPyramid.push_back(LeftGaussianPyramid[5]);
   //imshow("LeftLaplacianPyramid", LeftLaplacianPyramid.back());
   //waitKey(0);
   for (int i = 5; i > 0; i--) {
       Mat OneLevelUp;
       pyrUp(LeftGaussianPyramid[i], OneLevelUp);
       Mat CurrentLaplacianPyramid = LeftGaussianPyramid[i - 1] - OneLevelUp;
       LeftLaplacianPyramid.push_back(CurrentLaplacianPyramid);
       //imshow("LeftLaplacianPyramid", LeftLaplacianPyramid.back());
       //waitKey(0);
   }
   //destroyAllWindows();

   
   //namedWindow("RightLaplacianPyramid", WINDOW_AUTOSIZE);
   //Create Laplacian Pyramid for the right image.
   vector<Mat_<Vec3f> > RightLaplacianPyramid;
   RightLaplacianPyramid.push_back(RightGaussianPyramid[5]);
   //imshow("RightLaplacianPyramid", RightLaplacianPyramid.back());
   //waitKey(0);
   for (int i = 5; i > 0; i--) {
       Mat OneLevelUp;
       pyrUp(RightGaussianPyramid[i], OneLevelUp);
       Mat CurrentLaplacianPyramid = RightGaussianPyramid[i - 1] - OneLevelUp;
       RightLaplacianPyramid.push_back(CurrentLaplacianPyramid);
       //imshow("RightLaplacianPyramid", RightLaplacianPyramid.back());
       //waitKey(0);
   }
   //destroyAllWindows();


   //Create a BlendMask that has 1 for the left half of the image and 0 for the right half of the image.
   Mat_<float> BlendMask(LeftImage.rows, LeftImage.rows, 0.0);
   BlendMask(Range::all(), Range(0, BlendMask.cols / 2)) = 1.0;

   //Create a Gaussian Pyramid for the BlendMask to be able to blend all the Laplacian Pyramids of the left and right images.
   //namedWindow("BlendMaskGaussianPyramid", WINDOW_AUTOSIZE);
   vector<Mat_<Vec3f> > BlendMaskGaussianPyramid;
   cvtColor(BlendMask, CurrentImage, CV_GRAY2BGR);
   BlendMaskGaussianPyramid.push_back(CurrentImage);
   //imshow("BlendMaskGaussianPyramid", CurrentImage);
   //waitKey(0);
   for (int i = 1; i < 6; i++) {
      Mat OneLevelDown;
      pyrDown(CurrentImage, OneLevelDown);
      BlendMaskGaussianPyramid.push_back(OneLevelDown);
      CurrentImage = OneLevelDown;
      //imshow("BlendMaskGaussianPyramid", BlendMaskGaussianPyramid.back());
      //waitKey(0);
   }
   //destroyAllWindows();


   //namedWindow("ResultLaplacianPyramid", WINDOW_AUTOSIZE);
   //Construct a Laplacian Pyramid mixing all the left and right images at the centre.
   vector<Mat_<Vec3f> > ResultLaplacianPyramid;
   for (int i = 0; i < 6; i++) {
      Mat LeftHalf = LeftLaplacianPyramid[i].mul(BlendMaskGaussianPyramid[5 - i]);
      Mat RightHalf = RightLaplacianPyramid[i].mul(Scalar(1.0, 1.0, 1.0) - BlendMaskGaussianPyramid[5 - i]);
      Mat BlendedLaplacianImage = LeftHalf + RightHalf;
      ResultLaplacianPyramid.push_back(BlendedLaplacianImage);
      //imshow("ResultLaplacianPyramid", BlendedLaplacianImage);
      //waitKey(0);
   }
   //destroyAllWindows();


   //namedWindow("FinalImage", WINDOW_AUTOSIZE);
   //Reconstruct the FinalImage from all the BlendedLaplacianPyramid.
   CurrentImage = ResultLaplacianPyramid[0];
   for (int i = 1; i < 6; i++) {
      pyrUp(CurrentImage, CurrentImage);
      CurrentImage += ResultLaplacianPyramid[i];
      //imshow("FinalImage", CurrentImage);
      //waitKey(0);
   }

   convertScaleAbs(CurrentImage, img_out, 255.0);

   return img_out;
}

bool Question3(char* argv[])
{
   // Read in input images
   Mat input_image1 = imread(argv[2], IMREAD_COLOR);
   Mat input_image2 = imread(argv[3], IMREAD_COLOR);

   // Histogram equalization
   Mat output_image = laplacian_pyramid_blending(input_image1, input_image2);

   // Write out the result
   string output_name = string(argv[4]) + string("output3.jpg");
   imwrite(output_name.c_str(), output_image);

   return true;
}

int main(int argc, char* argv[])
{
   int question_number = -1;

   // Validate the input arguments
   if (argc < 4) {
      help_message(argv);
      exit(1);
   }
   else {
      question_number = atoi(argv[1]);

      if (question_number == 1 && !(argc == 4)) {
         help_message(argv);
         exit(1);
      }
      if (question_number == 2 && !(argc == 5)) {
	 help_message(argv);
	 exit(1);
      }
      if (question_number == 3 && !(argc == 5)) {
	 help_message(argv);
	 exit(1);
      }
      if (question_number > 3 || question_number < 1 || argc > 5) {
	 cout << "Input parameters out of bound ..." << endl;
	 exit(1);
      }
   }

   switch (question_number) {
      case 1: Question1(argv); break;
      case 2: Question2(argv); break;
      case 3: Question3(argv); break;
   }

   return 0;
}
