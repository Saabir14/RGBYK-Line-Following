import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def process_symbol_recognition(image, lowHSV, highHSV, source):
    """
    Process symbol recognition on an image.

    Args:
        image (MatLike): The input image.
        lowHSV (tuple): The lower HSV threshold for color masking.
        highHSV (tuple): The upper HSV threshold for color masking.
        source (list): List of source images for symbol comparison.

    Returns:
        None
    """

    # Mask color
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), lowHSV, highHSV)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxed_image = image.copy()

    # Iterate over the contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the polygon has 4 vertices, it is a square
        if len(approx) == 4:
            # Draw the contour on the image
            cv2.drawContours(boxed_image, [approx], -1, (0, 255, 0), 2)

    # Define the size of the output image
    # hight needs to be the same as the original image if intended to be used with np.hstack()
    output_size = image.shape[0]
    
    # Iterate over the contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the polygon has 4 vertices, it is a square
        if len(approx) == 4:
            # Draw the contour on the image
            cv2.drawContours(boxed_image, [approx], -1, (0, 255, 0), 2)

            # Sort the corners of the contour
            approx = approx.reshape((4, 2))
            approx = approx[np.argsort(approx[:, 1])]

            if approx[0, 0] > approx[1, 0]:
                approx[[0, 1]] = approx[[1, 0]]

            if approx[2, 0] < approx[3, 0]:
                approx[[2, 3]] = approx[[3, 2]]

            # Define the corners of the output image
            output_corners = np.array([[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]], dtype='float32')

            # Compute the transformation matrix
            M = cv2.getPerspectiveTransform(approx.astype('float32'), output_corners)

            # Apply the perspective transformation
            warped = cv2.warpPerspective(mask, M, (output_size, output_size))

    # Use image recognition to find the symbol that the warped image most looks like from the source
    similarities = []
    for symbol in source:
        # Convert the symbol to HSV
        maskedSymbol = cv2.inRange(cv2.cvtColor(symbol, cv2.COLOR_BGR2HSV), lowHSV, highHSV)
        similarities.append(max(ssim(warped, maskedSymbol := cv2.rotate(maskedSymbol, cv2.ROTATE_90_CLOCKWISE)) for _ in range(4)))
    
    most_similar_image = source[np.argmax(similarities)]
    
    # Display useful debug info
    # ---
    cv2.imshow("Debug", np.hstack((image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), boxed_image, cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR), most_similar_image)))
    key = cv2.waitKey(0)

    # If the 'ESC' key is pressed, close all debug windows
    if key == 27:
        cv2.destroyAllWindows()
    # ---

    return most_similar_image

if __name__ == "__main__":
    # Test function
    image = cv2.imread("distorted_symbols/Umbrella (Yellow Line).png")
    source_file = "symbols"

    images = [cv2.imread(os.path.join("symbols", image_file_path)) for image_file_path in os.listdir(source_file) if cv2.imread(os.path.join(source_file, image_file_path)) is not None]
    process_symbol_recognition(image, (140, 100, 100), (180, 255, 255), images)