import cv2
import numpy as np
from smbus2 import SMBus
import struct

import os
from symbol_recognition import symbol_recognition

# Initialize the I2C bus.
bus = SMBus(1)

# Define the device address.
device_address = 0x04

# Define the color boundaries.
color_bounds = {
    'r': (np.array([0, 120, 20]), np.array([7, 255, 255]), (np.array([176, 102, 20]), np.array([180, 255, 255]))),  # red
    'g': (np.array([35, 102, 20]), np.array([80, 255, 255])),  # green
    'b': (np.array([89, 102, 20]), np.array([125, 255, 255])),  # blue
    'y': (np.array([17, 102, 20]), np.array([34, 255, 255])),  # yellow
    'k': (np.array([0, 0, 0]), np.array([180, 102, 120])),  # black
    'p': (np.array([100, 40, 0]), np.array([160, 255, 255])), # pink
}

# Load the images for symbol recognition.
source_file = "symbols"
images = {}
for image_file_path in os.listdir(source_file):
    image = cv2.imread(os.path.join(source_file, image_file_path))
    if image is not None:
        images[image_file_path] = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (100, 40, 0), (160, 255, 255))

symbol_detection_threshold = .2

def RGBYK_line_following():
    # Initialize the camera.
    cap = cv2.VideoCapture(0)

    # Ask the user to choose a color.
    color = 'k'

    # Create a weight matrix where each column is filled with the column index minus half the frame width.
    weights = np.arange(-cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
    total = np.sum(weights)

    while True:
        # Capture a frame.
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture a frame.")
            continue

        # Convert the frame to HSV.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Check if symbol recognition is needed.
        pink_mask = np.bitwise_or.reduce([cv2.inRange(hsv, np.array(lower), np.array(upper)) for lower, upper in color_bounds['p']])
        if np.sum(pink_mask) / (255 * pink_mask.size) > symbol_detection_threshold:
            color = symbol_recognition(frame, images, .75)

        # Apply the color bounds.
        mask = np.bitwise_or.reduce([cv2.inRange(hsv, np.array(lower), np.array(upper)) for lower, upper in color_bounds[color]])

        # Find the weighted average.
        # w = np.sum(mask[mask != 0] * weights[mask != 0]) / total
        w = np.dot(mask[mask != 0], weights[mask != 0]) / total

        # Write the weighted average to the device.
        bus.write_i2c_block_data(device_address, 0, struct.pack('d', w))
        ''' If error occurs, such as bytes sent the wrong way, try the following code instead:
        bus.write_i2c_block_data(device_address, 0, list(struct.pack('<d', w)))
            OR
        bus.write_i2c_block_data(device_address, 0, [ord(c) for c in struct.pack('>d', w)])
        '''

        ''' Arduino code to receive and build w
        #include <Wire.h>

        #define DEVICE_ADDRESS 0x04
        #define DATA_LENGTH 8

        double w;

        void setup() {
            Wire.begin(DEVICE_ADDRESS); // join i2c bus with address #4
            Wire.onReceive(receiveEvent); // register event
            Serial.begin(9600); // start serial for output
        }

        void loop() {
            delay(100);
        }

        // function that executes whenever data is received from master
        void receiveEvent(int howMany) {
            if (howMany == DATA_LENGTH) {
                uint8_t buffer[DATA_LENGTH];
                for (int i = 0; i < DATA_LENGTH; i++) {
                    buffer[i] = Wire.read();
                }
                w = *((double*)buffer);
                Serial.println(w);
            }
        }
        '''
        # if condition_to_break: break

    # Release the camera and close all windows.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    RGBYK_line_following()