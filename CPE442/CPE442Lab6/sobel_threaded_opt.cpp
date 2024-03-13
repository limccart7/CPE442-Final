/*

Liam McCarthy, Wyatt Colburn
CPE442 Real Time Embedded Systems
Lab 6 - Speedup, performance metrics

Final project to speed up and compare results of sobel filter

Things Added:
1. Barriers, removed the need to join threads in the main loop,
only joining after last frame is displayed

2. Floating point approximation for grayscale calculation

3. Vectorization of grayscale

4. Comparison to -O3 compiler flag

Header file - contains all functions and logic

*/


#include "sobel_threaded.hpp"

struct sobel_thread_args
{
    pthread_t* thread; //pointer to specific thread
    Mat* inMat; //pointer to input frame
    Mat* grayMat; //pointer to shared grayscale mat
    Mat* outMat;  //shared output mat

    int thread_id; //specify which thread

    int start_row; //specify where to start and end ops
    int end_row; //calculated from thread id

    uint8_t do_work;
};
//pthreads stuff
pthread_t sobelThreads[NUM_THREADS];
pthread_barrier_t outputReady, newframeReady;

//approximate weights for grayscale vectorization, 256 so we can right shift 8 times
const uint8_t GRAY_WEIGHTS[24] = {
    (uint8_t)(.0722*256), (uint8_t)(.7152*256), (uint8_t)(.2126*256), 
    (uint8_t)(.0722*256), (uint8_t)(.7152*256), (uint8_t)(.2126*256),
    (uint8_t)(.0722*256), (uint8_t)(.7152*256), (uint8_t)(.2126*256),
    (uint8_t)(.0722*256), (uint8_t)(.7152*256), (uint8_t)(.2126*256),
    (uint8_t)(.0722*256), (uint8_t)(.7152*256), (uint8_t)(.2126*256),
    (uint8_t)(.0722*256), (uint8_t)(.7152*256), (uint8_t)(.2126*256),
    (uint8_t)(.0722*256), (uint8_t)(.7152*256), (uint8_t)(.2126*256),
    (uint8_t)(.0722*256), (uint8_t)(.7152*256), (uint8_t)(.2126*256)
    };
//sobel operation function
void* sobel_thread(void* sobelArgs){
    struct sobel_thread_args *threadArgs = (struct sobel_thread_args *)sobelArgs; //cast back
    int start_row, end_row, img_row, img_col;
    if (threadArgs->thread_id == 0){ //top thread, clamp first row so it stays in bounds
        start_row = 1;
    }
    else{
        start_row = threadArgs->start_row;
    }
    if (threadArgs->thread_id == 3){ //bottom thread, clamp last row
        end_row = threadArgs->outMat->rows - 1;
    } 
    else{
        end_row = threadArgs->outMat->rows;
    }   

    int16x8_t TL, ML, BL, TM, BM, TR, MR, BR, sobelWvect;
    long frame_pixels = threadArgs->inMat->rows * threadArgs->inMat->cols;
    long pix;
    uint8_t cleanup = frame_pixels % 8; //leftover pixels, if any
    uint8x8x3_t temp_rgb, gray_weights;
    uint8x16x3_t temp_val;
    uint8x8_t results;
    uint8_t* in_ptr, gray_ptr, gray_top_row, gray_mid_row, gray_bottom_row, output;

    while(threadArgs->do_work){
        pthread_barrier_wait(&newframeReady); //wait until new frame is in

        in_ptr = threadArgs->inMat->ptr<uchar>(threadArgs->start_row);
        gray_ptr = threadArgs->grayMat->ptr<uchar>(threadArgs->start_row);
        frame_pixels = threadArgs->inMat->rows * threadArgs->inMat->cols;
        gray_weights = vld3_u8(&GRAY_WEIGHTS); 

        for(pix = 0; pix < frame_pixels - 7; pix += 8){
            results = vdup_n_u8(0); //clear vals
            temp_rgb = vld3_u8(in_ptr); //load 3 vectors for RGB
            for(uint8_t i = 0; i < 3; i++){
                temp_val[i] = vmull_u8(temp_rgb[i], gray_weights[i]); //widening multiply
                temp_val[i] = vshrn_u16(temp_val[i]); //divide back by 256
                results = vadd_u8(results, vmovn_u16(temp_val[i])); //add to gray results
            }
            vst1_u8(gray_ptr, results);
            in_ptr += 8;
            gray_ptr += 8;
        }

        //cleanup for last pixels if any
        for(pix 0; pix < cleanup; pix++){
            uint8_t grayVal = ((0.2126 * in_ptr[2]) + (0.7152 * in_ptr[1]) + (0.0722 * in_ptr[0]));
            *gray_ptr = grayVal; //put the value in..?
            gray_ptr++;
            in_ptr++;
        }
    
        //grayscale mat should be shared, can grab pixel values no problem
        for(img_row = start_row; img_row < end_row; img_row++){ //iterate row
            gray_mid_row = threadArgs->grayMat->ptr<uchar>(img_row); //ptr to first pixel in current row
            gray_bottom_row = threadArgs->grayMat->ptr<uchar>(img_row + 1); //ptr to first pixel in below row
            gray_top_row = threadArgs->grayMat->ptr<uchar>(img_row - 1);
            output = threadArgs->outMat->ptr<uchar>(img_row); //output row pointer
            for (img_col = 1; img_col < (threadArgs->grayMat->cols - 7); img_col += 8){ //iterate cols

                TL = vmovl_s8(vld1_s8(gray_top_row)); //load 8 pixels into size 8 vectors, then make all vectors size 16 without sign extend
                TM = vmovl_s8(vld1_s8(gray_top_row + 1)); //top mid
                TR = vmovl_s8(vld1_s8(gray_top_row + 2)); //top right
                ML = vmovl_s8(vld1_s8(gray_mid_row)); //middle left
                MR = vmovl_s8(vld1_s8(gray_mid_row + 2)); //middle right
                BL = vmovl_s8(vld1_s8(gray_bottom_row)); //bottom left
                BM = vmovl_s8(vld1_s8(gray_bottom_row + 1)); //bottom mid
                BR = vmovl_s8(vld1_s8(gray_bottom_row + 2)); //bottom right

                sobelWvect = vaddq_s16( \
                vabsq_s16( //absolute of X sobel operation
                vsubq_s16(vaddq_s16(vaddq_s16(MR, MR), vaddq_s16(TR, BR)), vaddq_s16(vaddq_s16(ML, ML), vaddq_s16(TL, BL)))), \
                vabsq_s16( //absolute of sobel Y operation
                vsubq_s16(vaddq_s16(vaddq_s16(TM, TM), vaddq_s16(TR, TL)), vaddq_s16(vaddq_s16(BM, BM), vaddq_s16(BR, BL)))));

                vst1_s8((output + 1), vmovn_s16(sobelWvect)); //stores the 8 values in a row
                gray_mid_row += 8;
                gray_top_row += 8;
                gray_bottom_row += 8;
                output += 8;
            }
        }
        pthread_barrier_wait(&outputReady); //output is ready, unlock main thread
    }
    return NULL;
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
        thread_args[i].thread_id = i;
        thread_args[i].start_row = (i * inMat.rows) / NUM_THREADS;
        thread_args[i].end_row = ((i + 1) * inMat.rows) / NUM_THREADS;
        thread_args[i].do_work = 1;
    }
    
    //init barriers
    pthread_barrier_init(&outputReady, NULL, NUM_THREADS + 1);
    pthread_barrier_init(&newframeReady, NULL, NUM_THREADS + 1);
    
    for(int i = 0; i < NUM_THREADS; i++){ //start the threads
        pthread_create(&sobelThreads[i], nullptr, sobel_thread, &thread_args[i]);
    }

    auto start = chrono::steady_clock::now(); //get start time
    long frames = 0;

    while(1){ //not done with code to actually setup and take down the threads
        cap >> inMat; //put new frame in
        if(inMat.empty()){
            break;
        }
        pthread_barrier_wait(&newframeReady); //new frame is in let the threads cook
        pthread_barrier_wait(&outputReady); //display, check for new frame
        
        imshow(title, outMat);

        frames+=1;
        if (waitKey(1) >= 0) {
            break;
        }
    }

    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    double fps = static_cast<double>(frameCount) / (duration / 1000.0);

    cout << "Averaged per second: " << fps << endl;

    //destroy
    for(int i = 0; i < NUM_THREADS; i++){ //wait for them to finish
        thread_args[i].do_work = 0; //break out of loop
        pthread_join(sobelThreads[i], NULL); //collapse threads
    }
    pthread_barrier_destroy(&outputReady, NULL, NUM_THREADS + 1);
    pthread_barrier_destroy(&newframeReady, NULL, NUM_THREADS + 1);
    
}
