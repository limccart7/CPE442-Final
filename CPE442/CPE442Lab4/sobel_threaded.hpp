/*

Liam McCarthy, Wyatt Colburn
CPE442 Real Time Embedded Systems
Lab 4 - Threaded Sobel

Uses multithreading to speed up a simple sobel filter on a raspberry pi

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
using namespace cv;
using namespace std;

#define NUM_THREADS 4

int sobel_filter_threaded(string videoName);

#endif


