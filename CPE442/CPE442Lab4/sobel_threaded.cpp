/*

Liam McCarthy, Wyatt Colburn
CPE442 Real Time Embedded Systems
Lab 4 - Threaded Sobel

Uses multithreading to speed up a simple sobel filter on a raspberry pi

Header file - contains all functions and logic used to multithread

*/


#include "sobel_threaded.hpp"

struct sobel_thread_args
{
    pthread_t* thread; //pointer to specific thread
    Mat* inMat; //pointer to input frame
    Mat* grayMat; //pointer to shared grayscale mat
    Mat* outMat;  //shared output mat

    int start_row; //specify where to start and end ops
    int end_row; //calculated from thread id
};

//init thread markers
//static pthread_t op_thread[4]; //4 child threads for math
//static pthread_t vid_thread;
//static pthread_barrier_t readyFrame, sobelDone; //barriers to wait for frame ready and sobel ops done


//sobel operation function
void* sobel_thread(void* sobelArgs){
    struct sobel_thread_args *threadArgs = (struct sobel_thread_args *)sobelArgs; //cast back

    for(int img_row = threadArgs->start_row; img_row < threadArgs->end_row; img_row++){ //iterate rows
            for (int img_col = 0; img_col < threadArgs->inMat->cols; img_col++){ //iterate cols
                uint8_t *colors = threadArgs->inMat->ptr(img_row, img_col);//grab color data at that pixel

                //grayscale calc
                uint8_t grayVal = ((0.2126 * colors[2]) + (0.7152 * colors[1]) + (0.0722 * colors[0]));

                //put grayscale value into the grayscale mat at the right place
                threadArgs->grayMat->at<uchar>(img_row, img_col) = uchar(grayVal);
            }
        }
    
    uint8_t TL, ML, BL, TM, BM, TR, MR, BR;
    //grayscale mat should be shared, can grab pixel values no problem
    for(int img_row = threadArgs->start_row; img_row < threadArgs->end_row; img_row++){ //iterate rows
            for (int img_col = 0; img_col < threadArgs->grayMat->cols; img_col++){ //iterate cols

                TL = threadArgs->grayMat->at<uchar>(img_row - 1,img_col - 1); //top left
                ML = threadArgs->grayMat->at<uchar>(img_row,img_col - 1); //middle left
                BL = threadArgs->grayMat->at<uchar>(img_row + 1,img_col - 1); //bottom left
                TM = threadArgs->grayMat->at<uchar>(img_row - 1,img_col); //top middle
                BM = threadArgs->grayMat->at<uchar>(img_row + 1,img_col); //bottom middle
                TR = threadArgs->grayMat->at<uchar>(img_row - 1,img_col + 1); //top right
                MR = threadArgs->grayMat->at<uchar>(img_row,img_col + 1); //middle right
                BR = threadArgs->grayMat->at<uchar>(img_row + 1,img_col + 1); //bottom right

                uint8_t sobelWeight = \
                abs((2 * MR) + BR + TR - (2 * ML) - BL - TL) + //sobel X operation
                abs((2 * TM) + TL + TR - (2 * BM) - BL - BR); //sobel y
                
                threadArgs->outMat->at<uchar>(img_row, img_col) = sobelWeight; //put calculated value into the mat
            }
    }
    return (void *) sobelArgs;
}


//manages a 1 parent 4 child thread version of the previous sobel filter function
//creates struct threads, divides screen up by 4 rows, 
int sobel_filter_threaded(string videoName){
    VideoCapture cap(videoName);

    if (!cap.isOpened()){ //no video exists
        cout << "Failed to Open Video. Did you type the name right??" << endl; //Miguel
        exit(-1);
    }
    //make shared mats and stuff
    Mat inMat;
    pthread_t sobelThreads[NUM_THREADS];
    struct sobel_thread_args thread_args[NUM_THREADS];
    cap >> inMat; //put first frame in
    Mat grayMat(inMat.size(), CV_8UC1);
    Mat outMat(inMat.size(), CV_8UC1);

    string title = "Threaded Sobel: " + videoName;
    namedWindow(title, WINDOW_NORMAL);

    //init child thread structs
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i].thread = &sobelThreads[i];
        thread_args[i].inMat = &inMat;
        thread_args[i].outMat = &outMat;
        thread_args[i].grayMat = &grayMat;
        thread_args[i].start_row = (i * inMat.rows) / NUM_THREADS;
        thread_args[i].end_row = ((i + 1) * inMat.rows) / NUM_THREADS;
    }
    
    while(1){ //not done with code to actually setup and take down the threads
        if(inMat.empty()){
            break;
        }
        for(int i = 0; i < NUM_THREADS; i++){ //start the threads
            pthread_create(&sobelThreads[i], nullptr, sobel_thread, &thread_args[i]);
        }
        for(int i = 0; i < NUM_THREADS; i++){ //wait for them to finish
            pthread_join(sobelThreads[i], NULL);
        }
        imshow(title, outMat);
        waitKey(1);


        cap >> inMat; //put new frame in
    }

    return 0;
    
}
