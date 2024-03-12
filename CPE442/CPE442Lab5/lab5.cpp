/*

Liam McCarthy, Wyatt Colburn
CPE442 Real Time Embedded Systems
Lab 5 - Vectorized/Parallelized, Threaded Sobel

Uses ARM Neon vecotr intrinsics to speed up a simple sobel
filter on a raspberry pi

Lab C++ file - runs the program

*/
#include "sobel_threaded.hpp"

int main(int argc, char *argv[]){
    if (argc != 2){
        printf("Usage: ./lab5 <filename>.mp4");
        return -1;
    }
    sobel_filter_threaded(argv[1]);
    return 0;
}