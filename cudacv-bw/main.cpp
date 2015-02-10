#include <opencv2/opencv.hpp>
#include <iostream>
#include "utils.h"
#include <string>
#include <stdio.h>

// Function calling the kernel to operate
void rgba_to_grey(uchar4 * const d_rgbaImage,
                  unsigned char* const d_greyImage, 
                  size_t numRows, size_t numCols);

//include the definitions of the above functions for this homework
#include "preprocess.cpp"

int main(int argc, char **argv) {
  uchar4        *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

  std::string input_file;
  std::string output_file;

  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  switch (argc)
  {
	case 2:
	  input_file = std::string(argv[1]);
	  output_file = "output.png";
	  break;
	default:
      std::cerr << "Usage: ./to_bw input_file [output_filename]" << std::endl;
      exit(1);
  }
  //load the image and give us our input and output pointers
  preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

  //call the cuda code
  rgba_to_grey(d_rgbaImage, d_greyImage, numRows(), numCols());

  size_t numPixels = numRows()*numCols();
  checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

  /* Output the grey image */
  cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)h_greyImage);
  // Open the window
  cv::namedWindow("to_bw");
  // Display the image m in this window
  cv::imshow("to_bw", output);
  cvWaitKey (0);
  cvDestroyWindow ("to_bw");
  //output the image
  cv::imwrite(output_file.c_str(), output);

  /* Cleanup */
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);

  return 0;
}
