/*

Liam McCarthy, Wyatt Colburn
CPE442 Real Time Embedded Systems
Lab 6 - Speedup, performance metrics

Final project to speed up and compare results of sobel filter

Header file - contains all includes, namespaces and
function prototypes

*/
#ifndef SOBEL_HPP
#define SOBEL_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <pthread.h>
#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <arm_neon.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

#define NUM_THREADS 4

int sobel_filter_threaded(string videoName);
void* sobel_thread(void* sobelArgs);

#endif

