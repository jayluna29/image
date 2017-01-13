#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;


__global__ void addKernel(unsigned char * in, unsigned char * out)
{						 //usually you add arrays with a for loop
	int x = blockIdx.x;
	int y = threadIdx.x;
	int width = blockDim.x;

	int index = (x + y * width) * 4;

	//copy each color channel
	out[index] = in[index];
	out[index + 1] = in[index + 1];
	out[index + 2] = in[index + 2];
	out[index + 3] = in[index + 3];


}


void threashold(int threashold, int width, int height, unsigned char *data);
int main(int argc, char ** argv)
{
	if (argc != 2)
	{
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	Mat image;

	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if (!image.data)
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	cvtColor(image, image, cv::COLOR_RGB2GRAY);
	//cout << "Converted to gray" << endl;
	threashold(220, image.cols, image.rows, image.data);

	//*********
	const unsigned int w = image.cols;
	const unsigned int h = image.rows;
	int size = w * h * 4;
	cudaError_t cudaStatus;
	unsigned char * in = 0;
	unsigned char * out = 0;

	// Allocate GPU buffers for three vectors (two input, one output)    
	cudaStatus = cudaMalloc((void**)&in, sizeof(unsigned char)); //mallocs in dev memory
	cudaStatus = cudaMalloc((void**)&out, sizeof(unsigned char)); //mallocs in dev memory

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(in, &image.data[0], size * sizeof(unsigned char), cudaMemcpyHostToDevice);

	copy <<<w, h>>> (in, out);

	cudaStatus = cudaDeviceSynchronize();
	//*********

	namedWindow("Display window", WINDOW_NORMAL);
	imshow("Display window", image);

	waitKey(0);

	return 0;
}
void threashold(int threashold, int width, int height, unsigned char *data)
{
	for (int i = 0; i < (height*width); i++)
	{
		if (data[i] > threashold)
		{
			data[i] = 255;
		}
		else
		{
			data[i] = 0;
		}
	}
}

