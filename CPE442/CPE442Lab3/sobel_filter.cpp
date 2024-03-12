/*

Liam McCarthy, Wyatt Colburn
CPE442 Real Time Embedded Systems
Lab 3 - Simple Sobel Filter

Takes a video file name as an input and converts it
to grayscale, then applies the sobel filter function
on each pixel. It displays the original, grayscale,
and sobel videos

Code File - Contains functions for the lab

*/

#include "sobel_filter.hpp"
//main function for sobel, takes video name as input
//displays the original video, the grayscale conversion
//and the sobel filtered version in new windows

int sobel_filter_vid(string filename){
    VideoCapture cap(filename);

    //declare vars for filters
    unsigned char* colors;
    int sobelWeight, gray, TL, TM, TR, ML, MR, BL, BM, BR; //variables used for ops

    if (!cap.isOpened()){ //no video exists
        cout << "Failed to Open Video. Did you type the name right??" << endl; //Miguel
        exit(-1);
    }

    //make windows and names
    string title1 = "Playing Original: " + filename;
    string title2 = "Playing Grayscale: " + filename;
    string title3 = "Playing Sobel: " + filename;
    namedWindow(title1, WINDOW_NORMAL);
    namedWindow(title2, WINDOW_NORMAL);
    namedWindow(title3, WINDOW_NORMAL);

    while(1){//continuously update windows

        //capture frame
        Mat frame;
        cap >> frame;

        if (frame.empty()){
            break;
        }

        Mat grayMat(frame.size(), CV_8UC1); //new mat to hold calculated grascale, only 1 val per pixel


        for(int img_row = 0; img_row < frame.rows; img_row++){ //iterate rows
            for (int img_col = 0; img_col < frame.cols; img_col++){ //iterate cols
                colors = frame.ptr(img_row, img_col);//grab color data at that pixel

                //grayscale calc
                gray = ((0.2126 * colors[2]) + (0.7152 * colors[1]) + (0.0722 * colors[0]));

                //put grayscale value into the grayscale mat at the right place
                grayMat.at<uchar>(img_row, img_col) = uchar(gray);
            }
        }
        
        Mat sobelMat(frame.size(), CV_8UC1); //vector size is 1, also black/white image

        for(int img_row = 0; img_row < frame.rows; img_row++){ //iterate rows
            for (int img_col = 0; img_col < frame.cols; img_col++){ //iterate cols

                TL = grayMat.at<uchar>(img_row - 1,img_col - 1); //top left
                ML = grayMat.at<uchar>(img_row,img_col - 1); //middle left
                BL = grayMat.at<uchar>(img_row + 1,img_col - 1); //bottom left
                TM = grayMat.at<uchar>(img_row - 1,img_col); //top middle
                BM = grayMat.at<uchar>(img_row + 1,img_col); //bottom middle
                TR = grayMat.at<uchar>(img_row - 1,img_col + 1); //top right
                MR = grayMat.at<uchar>(img_row,img_col + 1); //middle right
                BR = grayMat.at<uchar>(img_row + 1,img_col + 1); //bottom right

                sobelWeight = \
                abs((2 * MR) + BR + TR - (2 * ML) - BL - TL) + //sobel X operation
                abs((2 * TM) + TL + TR - (2 * BM) - BL - BR); //sobel y
                
                sobelMat.at<uchar>(img_row, img_col) = sobelWeight; //put calculated value into the mat
            }
        }

        //put frame mat into windows
        imshow(title1, frame);
        moveWindow(title1, 50, 100);

        imshow(title2, grayMat);
        moveWindow(title2, 150, 100);

        imshow(title3, sobelMat);
        moveWindow(title3, 250, 100);

        waitKey(25);
    
    }
    cap.release();
    destroyAllWindows();

    return 0;
}