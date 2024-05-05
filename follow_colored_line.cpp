// Include files for required libraries
#include <stdio.h>
#include <iostream>
#include <pigpio.h>//needs to be included to allow use of pigpio functions
                   //among other features, allows for i2c communication
using namespace std;

#include "opencv_aee.hpp"
#include "main.hpp"     // You can use this file for declaring defined values and functions

void setup(void)
{
    setupCamera(320, 240);  // Enable the camera for OpenCV
}

int main( int argc, char** argv )
{
    setup();    // Call a setup function to prepare IO and devices

    cv::namedWindow("Photo");   // Create a GUI window called photo
    int nonZero[8] = {0,0,0,0,0,0,0,0};//initialise array to hold number of non zero values found in snippet of camera
    int distFromZeroPoint[8] = {1, 40, 80, 120, 160, 200, 240, 280};//distance from zero point used for weighted average calc
                                                        //originally set to pixel width, but that resulted in a value
                                                        //that was too large to send, so scaled down
    float weightedAverage = 0;//declare weighted average variable
    uint8_t weighAve = 0;//declare variable to store weighted average to be sent to esp32

    gpioInitialise();
    if (gpioInitialise() < 0)//if statement used to check whether initialisation was successful
    {
        //pigpio initialisation successful
    }
    else
    {
        //initialisation not successful
    }

    unsigned int deviceHandle=i2cOpen(1,0x04,0);//initialise connection on bus 1 with device with address 0x04

    bool isRed;
    char chooseColour;//declare variable which will store user input
    Scalar upperBound, lowerBound;//decalre variable for storing the upper and lower bound used in inRange function
    printf("Select which colour you wish the car to follow\nb=Blue | k=Black | r=Red | y=Yellow | g=Green\n");
    scanf("%c",&chooseColour);//allow user to choose which colour they want to follow

    //          -------------------------------------
    //         | lower boundaries | upper boundaries | for colors
    //green    |   [35,102,20]    |   [80,255,255]   |
    //lower red|   [0,102,20]     |   [7,255,255]    |
    //upper red|   [176,102,20]   |   [180,255,255]  |
    //blue     |   [89,102,20]    |   [125,255,255]  |
    //yellow   |   [17,102,20]    |   [34,255,255]   |
    //black    |     [0,0,0]      |   [180,102,120]  |
    //          -------------------------------------

    switch(chooseColour)//switch case for determining which boundaries to use in inRange
    {
        case'b'://blue
            lowerBound = Scalar(89,102,20);
            upperBound = Scalar(125,255,255);
            break;

        case'k'://black
            lowerBound = Scalar(0,0,0);
            upperBound = Scalar(180,102,120);
            break;

        case'g'://green
            lowerBound = Scalar(35,102,20);
            upperBound = Scalar(80,255,255);
            break;

        case'y'://yellow
            lowerBound = Scalar(17,102,20);
            upperBound = Scalar(34,255,255);
            break;

        default:
            if (chooseColour='r')//red requires a slightly different process than other colours cause
                                 //it's present on both sides of the hue spectrum
                isRed=true;
            else
                isRed=false;
                printf("Error, choose a different colour.");
    }


    while(1)    // Main loop to perform image processing
    {
        Mat frame, frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8;//declare all needed frames
        float top=0, bottom=0;//initialise values for the top and bottom part of the weighted average equation

        while(frame.empty())
            frame = captureFrame(); // Capture a frame from the camera and store in a new matrix variable

        cvtColor(frame, frame, COLOR_BGR2HSV); //convert image to hsv

        if (isRed==true)
        {
            cout << "Red is";
            Mat framer;//declare frame for storing upper red vals
            inRange(frame,Scalar(0,120,20),Scalar(7,255,255),frame);//check lower end of hue
            inRange(frame,Scalar(176,102,20),Scalar(180,255,255),framer);//check upper end of hue
            frame=frame+framer;//combine both upper and lower boundaries

        }
        else{
            inRange(frame, lowerBound, upperBound,frame);//check each pixel in matrix whether its colour is within the
                                                         //given range, if it is then it will be white, else will be
                                                         //black (value of zero)

            //frame1
            frame1 = frame(Rect(0,0,40,1)); //first two values are coords of top left pixel, 3rd is width, 4th is height
                                            //crop image, each cropped image works as a sensor
                                            //"amplitude of sensor" taken as number of non zero values

            nonZero[0] = countNonZero(frame1);

            //frame2
            frame2 = frame(Rect(40,0,40,1)); //first two values are coords of top left pixel, 3rd is width, 4th is height

            nonZero[1] = countNonZero(frame2);

            //frame3
            frame3 = frame(Rect(80,0,40,1)); //first two values are coords of top left pixel, 3rd is width, 4th is height

            nonZero[2] = countNonZero(frame3);

            //frame4
            frame4 = frame(Rect(120,0,40,1)); //first two values are coords of top left pixel, 3rd is width, 4th is height

            nonZero[3] = countNonZero(frame4);

            //frame5
            frame5 = frame(Rect(160,0,40,1)); //first two values are coords of top left pixel, 3rd is width, 4th is height

            nonZero[4] = countNonZero(frame5);

            //frame6
            frame6 = frame(Rect(200,0,40,1)); //first two values are coords of top left pixel, 3rd is width, 4th is height

            nonZero[5] = countNonZero(frame6);

            //frame7

            frame7 = frame(Rect(240,0,40,1)); //first two values are coords of top left pixel, 3rd is width, 4th is height

            nonZero[6] = countNonZero(frame7);

            //frame8
            frame8 = frame(Rect(280,0,40,1)); //first two values are coords of top left pixel, 3rd is width, 4th is height

            nonZero[7] = countNonZero(frame8);
        }
        //calculate weighted average
        for (int i=0;i<8;i++)
        {
            printf("%d ",nonZero[i]);
            top=top+(distFromZeroPoint[i]*nonZero[i]);//calculate top part of weighted average equation
            bottom=bottom+nonZero[i];//calculate bottom part of weighted average equation
        }

        weightedAverage=top/bottom*0.91;//calculate weighted average, includes constant so weighted average value stays
                                        //under 255, needed when sending over i2c as only 8bit value sent at a time
        printf("\nWeighted Average: %f",weightedAverage);//print weighted average, mostly for testing
        weighAve = (uint8_t)weightedAverage;//create a 8bit unsigned integer version of weighted average to send over i2c
        printf("weighave: %d",weighAve);//print the 8bit uint version of weighted average
        //printf("\nChoose colour is %c",chooseColour);//used for testing
        printf("\n\n");

        i2cWriteByte(deviceHandle, weighAve);//write weighted average to connected device

        cv::imshow("Photo", frame); //Display the image in the window

        int key = cv::waitKey(1);   // Wait 1ms for a keypress (required to update windows)

        key = (key==255) ? -1 : key;    // Check if the ESC key has been pressed
        if (key == 27)
            break;
      }

    gpioTerminate();//terminate current connection
      closeCV();  // Disable the camera and close any windows

      return 0;
}