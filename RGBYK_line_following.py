import cv2
import numpy as np
from smbus2 import SMBus
import struct
import os
import time

from symbol_recognition import symbol_recognition

# Define the device address.
DEVICE_ADDRESS = 0x04

# Define the color boundaries.
COLOR_BOUNDS = {
    'r': (np.array([0, 120, 20]), np.array([7, 255, 255]), (np.array([176, 102, 20]), np.array([180, 255, 255]))),  # red
    'g': (np.array([35, 102, 20]), np.array([80, 255, 255])),  # green
    'b': (np.array([89, 102, 20]), np.array([125, 255, 255])),  # blue
    'y': (np.array([17, 102, 20]), np.array([34, 255, 255])),  # yellow
    'k': (np.array([0, 0, 0]), np.array([180, 102, 120])),  # black
    'p': (np.array([100, 40, 0]), np.array([160, 255, 255])), # pink
}

# Define the default color that will be followed if the threshold amount of that color is seen.
DEFAULT_COLOR = 'k'
DEFAULT_COLOR_THRESHOLD = .15

# Define the source folder.
SOURCE_FOLDER = "symbols"

# Ratio of pink needed to start symbol detection
SYMBOL_START_DETECTION_THRESHOLD = .1
# Crop the frame by a ratio from the top, right, left, and bottom before checking for symbol_start_detection_threshold.
SYMBOL_START_DETECTION_CROP = {'t': .1, 'r': .1, 'b': .1, 'l': .1}
# Min SSIM for successful symbol recognition
SYMBOL_DETECTION_THRESHOLD = .2
# Cooldown for symbol detection
SYMBOL_DETECTION_COOLDOWN = 5

def RGBYK_line_following() -> None:
    """
    Perform line following using RGBYK color detection.

    This function initializes the camera, loads symbol images for recognition,
    captures frames from the camera, applies color detection and symbol recognition,
    calculates the weighted average of the detected color, and writes the result to a device.

    Returns:
        None
    """
    # Load the images for symbol recognition.
    symbols = {}
    for image_file_path in os.listdir(SOURCE_FOLDER):
        symbol = cv2.imread(os.path.join(SOURCE_FOLDER, image_file_path))
        if symbol is not None:
            symbols[image_file_path] = cv2.inRange(cv2.cvtColor(symbol, cv2.COLOR_BGR2HSV), COLOR_BOUNDS['p'][0], COLOR_BOUNDS['p'][1])

    # Initialize the camera.
    cap = cv2.VideoCapture(0)

    # Start color.
    color = DEFAULT_COLOR

    # Create a weight matrix where each column is filled with the column index minus half the frame width.
    weights = np.arange(-cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
    total = np.sum(weights)

    # Capture a frame.
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture first frame.")
        return

    # Crop the frame by a ratio from the top, right, left, and bottom before checking for symbol_start_detection_threshold.
    height, width = frame.shape[:2]
    crop_height_start, crop_height_end = int(height * SYMBOL_START_DETECTION_CROP['t']), int(height * 1 - SYMBOL_START_DETECTION_CROP['b'])
    crop_width_start, crop_width_end = int(width * SYMBOL_START_DETECTION_CROP['l']), int(width * 1 - SYMBOL_START_DETECTION_CROP['r'])
    crop_region = np.index_exp[crop_height_start:crop_height_end, crop_width_start:crop_width_end]

    # Initialize the I2C bus.
    bus = SMBus(1)

    tick = time.time()
    while True:
        # Capture a frame.
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture a frame.")
            continue

        # Convert the frame to HSV.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Check if symbol recognition is needed.
        if time.time() - tick > SYMBOL_DETECTION_COOLDOWN:
            pink_mask = np.bitwise_or.reduce([cv2.inRange(hsv[crop_region], lower, upper) for lower, upper in COLOR_BOUNDS['p']])
            if cv2.countNonZero(pink_mask) / pink_mask.size > SYMBOL_START_DETECTION_THRESHOLD:
                if color := symbol_recognition(frame, symbols, SYMBOL_DETECTION_THRESHOLD):
                    tick = time.time()
            else:
                default_mask = np.bitwise_or.reduce([cv2.inRange(hsv, lower, upper) for lower, upper in COLOR_BOUNDS[DEFAULT_COLOR]])
                if cv2.countNonZero(default_mask) / default_mask.size > DEFAULT_COLOR_THRESHOLD:
                    color = DEFAULT_COLOR

        # Apply the color bounds.
        mask = np.bitwise_or.reduce([cv2.inRange(hsv, lower, upper) for lower, upper in COLOR_BOUNDS[color]])

        # Find the weighted average.
        # w = np.sum(mask[mask != 0] * weights[mask != 0]) / total
        w = np.dot(mask[mask != 0], weights[mask != 0]) / total

        # Write the weighted average to the device.
        bus.write_i2c_block_data(DEVICE_ADDRESS, 0, struct.pack('d', w))
        ''' If errors occur, such as bytes sent the wrong way, try the following code instead:
        bus.write_i2c_block_data(device_address, 0, struct.pack('<d', w))
            OR
        bus.write_i2c_block_data(device_address, 0, [ord(c) for c in struct.pack('>d', w)])
        '''

        ''' Arduino code to receive and build w as double:
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