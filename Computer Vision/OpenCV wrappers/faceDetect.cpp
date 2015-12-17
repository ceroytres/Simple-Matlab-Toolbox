#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\objdetect\objdetect.hpp>
#include<iostream>
#include "mex.h"


#define IM_IN prhs[0]
#define LOC_OUT plhs[0]


using namespace cv;
using namespace std;


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs != 7)  mexErrMsgTxt("Error:Too many/ too little inputs");

	if (!mxIsDouble(prhs[0])) mexErrMsgTxt("Error: Input needs to be double MatLab data type!");
	
	if (!mxIsChar(prhs[1])) mexErrMsgTxt("Error: Input needs to contain a string containing the xml database! (Type char)");

	if (mxGetM(prhs[1]) != 1) mexErrMsgTxt("Error String needs to row vector");
	
	if (!mxIsDouble(prhs[2])) mexErrMsgTxt("Error: Input needs a scaling factor(double)");


	if ((mxGetM(prhs[2]) != 1 || mxGetN(prhs[2]) != 1)) mexErrMsgTxt("Error: Scaling factor need to be a scalar");

	if (!mxIsDouble(prhs[3])) mexErrMsgTxt("Error: Input needs a min neighbors(double)");

	if ((mxGetM(prhs[3]) != 1 || mxGetN(prhs[3]) != 1)) mexErrMsgTxt("Error: min neighbors needs to be a scalar");


	if (!mxIsDouble(prhs[4])) mexErrMsgTxt("Error: Input needs a flag");

	if ((mxGetM(prhs[4]) != 1 || mxGetN(prhs[4]) != 1)) mexErrMsgTxt("Error: Flag needs to be a scalar");
	
	if (!mxIsDouble(prhs[5])) mexErrMsgTxt("Error: Input needs to contain a row min size vector of type double");

	if (mxGetM(prhs[5]) != 1 || mxGetN(prhs[5]) != 2) mexErrMsgTxt("Error needs to be 1x2 size min vector size");

	if (!mxIsDouble(prhs[6])) mexErrMsgTxt("Error: Input needs to contain a row max size vector of type double");

	if (mxGetM(prhs[6]) != 1 || mxGetN(prhs[6]) != 2) mexErrMsgTxt("Error needs to be 1x2 size max vector size");


	double *imPtr, *minSizePtr, *maxSizePtr;
	double scaleFactor = mxGetScalar(prhs[2]);
	int flag=(int)mxGetScalar(prhs[4]);
	int minN = (int)mxGetScalar(prhs[3]);
	char *input_buf;

	int R, C, r, c;
	R = mxGetM(IM_IN);
	C = mxGetN(IM_IN);

	input_buf = mxArrayToString(prhs[1]);
	
	minSizePtr = mxGetPr(prhs[5]);
	Size minSize=Size((int)minSizePtr[0],(int)minSizePtr[1]);

	maxSizePtr = mxGetPr(prhs[6]);
	Size maxSize = Size((int)maxSizePtr[0],(int)maxSizePtr[1]);

	Mat im = Mat::zeros(R, C, CV_64F);
	CascadeClassifier faceCascade;
	String face_cascadeFile = input_buf;
	vector<Rect> faceLoc;
// 
//     mexPrintf("%d,%d\n", (int)maxSizePtr[0], (int)maxSizePtr[1]);
//     mexPrintf("%d,%d\n", (int)minSizePtr[0], (int)minSizePtr[1]);
//     mexPrintf("%d\n", (int)flag);
//     mexPrintf("%d\n",CV_HAAR_DO_ROUGH_SEARCH|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT);
//     mexPrintf("%d\n", (int)minN);
//     mexPrintf("%lf\n", scaleFactor);
//     
    
    
	imPtr = mxGetPr(IM_IN);

	if (!faceCascade.load(face_cascadeFile))
	{
		mexPrintf("Error: xml face database not in current directory!\n");
	}


	for (r = 0; r < R; r++)
	{
		for (c = 0; c < C; c++) {
			im.at<double>(r, c) = imPtr[r + c*R];
		}
	}
	im = 255 * im;
	im.convertTo(im, CV_8U);

	faceCascade.detectMultiScale(im, faceLoc, scaleFactor, minN, 0 | flag, minSize, maxSize);

	int numFaces = faceLoc.size();
	double *LOC_ptr;
	LOC_OUT = mxCreateDoubleMatrix(numFaces, 4, mxREAL);
	LOC_ptr = mxGetPr(LOC_OUT);


	for (int l = 0; l < numFaces; l++)
	{
		LOC_ptr[l] = faceLoc[l].x;
		LOC_ptr[l + numFaces] = faceLoc[l].y;
		LOC_ptr[l + 2 * numFaces] = faceLoc[l].width;
		LOC_ptr[l + 3 * numFaces] = faceLoc[l].height;
	}


	return;
}


