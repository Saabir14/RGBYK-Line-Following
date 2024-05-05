import cv2
import numpy as np
from smbus2 import SMBus
import struct

# Initialize the I2C bus.
bus = SMBus(1)

# Define the device address.
device_address = 0x04

# Define the color boundaries.
color_bounds = {
    'r': (((0, 120, 20), (7, 255, 255)), ((176, 102, 20), (180, 255, 255))),  # red
    'g': ((35, 102, 20), (80, 255, 255)),  # green
    'b': ((89, 102, 20), (125, 255, 255)),  # blue
    'y': ((17, 102, 20), (34, 255, 255)),  # yellow
    'k': ((0, 0, 0), (180, 102, 120)),  # black
}

def RGBYK_line_following():
    # Initialize the camera.
    cap = cv2.VideoCapture(0)

    # Ask the user to choose a color.
    choose_color = input("Select which colour you wish the car to follow\nb=Blue | k=Black | r=Red | y=Yellow | g=Green\n")

    # Create a weight matrix where each column is filled with the column index minus half the frame width.
    weights = np.arange(-cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
    total = np.sum(weights)

    while True:
        # Capture a frame.
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture a frame.")
            break

        # Convert the frame to HSV.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply the color bounds.
        mask = np.bitwise_or.reduce([cv2.inRange(hsv, np.array(lower), np.array(upper)) for lower, upper in color_bounds[choose_color]])

        # Calculate the sum of the product of the non-zero pixels in the mask and the weight matrix.
        sum = np.sum(mask[mask != 0] * weights[mask != 0])

        w = sum / total if total != 0 else 0

        # Write the weighted average to the device.
        bus.write_i2c_block_data(device_address, 0, struct.pack('d', w))
        '''
        If error occurs, such as bytes sent the wrong way, try the following code instead:
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

    # Release the camera and close all windows.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    RGBYK_line_following()