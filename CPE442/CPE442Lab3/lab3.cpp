/*

Liam McCarthy, Wyatt Colburn
CPE442 Real Time Embedded Systems
Lab 3 - Simple Sobel Filter

Takes a video file name as an input and converts it
to grayscale, then applies the sobel filter function
on each pixel. It displays the original, grayscale,
and sobel videos

Runs the sobel filter on a pre-selected video

*/

#include "sobel_filter.hpp"

int main(void){
    string filename = "ferb.mp4";
    sobel_filter_vid(filename);
    //filename = "fire.mp4";
    //sobel_filter_vid(filename);

    return 0;
}