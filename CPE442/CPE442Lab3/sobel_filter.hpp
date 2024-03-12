/*

Liam McCarthy, Wyatt Colburn
CPE442 Real Time Embedded Systems
Lab 3 - Simple Sobel Filter

Takes a video file name as an input and converts it
to grayscale, then applies the sobel filter function
on each pixel. It displays the original, grayscale,
and sobel videos

Header file - contains all includes, namespaces and
function prototypes

*/
#ifndef SOBEL_HPP
#define SOBEL_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>
using namespace cv;
using namespace std;

int sobel_filter_vid(string videoName);

#endif


