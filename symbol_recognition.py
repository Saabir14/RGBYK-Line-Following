import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def symbol_recognition(mask: cv2.Mat, source: dict, minSSIM=-1, debug=False) -> str:
    """
    Process symbol recognition on an image.

    Args:
        mask (cv2.Mat): The input mask.
        lowHSV (tuple): The lower HSV threshold for color masking.
        highHSV (tuple): The upper HSV threshold for color masking.
        source (dict): Dictionary of source images for symbol comparison. All images should be the same size and already masked.
        minSSIM (float): The minimum structural similarity index for a match. Default is -1.
        debug (bool): Whether to display debug information. Default is False.
        output (str): The desired output format. Options are "name", "image", or "both". Default is "both".

    Returns:
        tuple: The name of the symbol and the most similar image, if output is "both".
        str: The name of the symbol, if output is "name".
        cv2.Mat: The most similar image, if output is "image".
    """

    # Find contours in the mask and copy the image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If the polygon has 4 vertices, it is a square
    if len(approx) == 4:
        # Sort the corners of the contour
        approx = approx.reshape((4, 2))
        approx = approx[np.argsort(approx[:, 1])]

        if approx[0, 0] > approx[1, 0]: approx[[0, 1]] = approx[[1, 0]]
        if approx[2, 0] < approx[3, 0]: approx[[2, 3]] = approx[[3, 2]]

        # Define the corners of the output image
        output_size = list(source.values())[0].shape[0]
        output_corners = np.array([[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]], dtype='float32')

        # Compute the transformation matrix and apply the perspective transformation
        M = cv2.getPerspectiveTransform(approx.astype('float32'), output_corners)
        warped = cv2.warpPerspective(mask, M, (output_size, output_size))

    # Use image recognition to find the symbol that the warped image most looks like from the source
    similarities = {key: max(ssim(warped, cv2.rotate(symbol, (i*90)%360)) for i in range(4)) for key, symbol in source.items()}

    name = max(similarities, key=similarities.get)
    most_similar_image = source[name]
    
    # Display useful debug info
    if debug:
        boxed_image = mask.copy()
        boxed_image = cv2.cvtColor(boxed_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(boxed_image, [approx], -1, (0, 255, 0), 2)

        showPool = (mask, boxed_image, warped, most_similar_image)

        # Define a function to add black bars (padding) to an image
        def add_padding(img, target_size):
            h, w = img.shape[:2]
            top = bottom = (target_size[0] - h) // 2
            left = right = (target_size[1] - w) // 2
            if (target_size[0] - h) % 2 != 0: bottom += 1
            if (target_size[1] - w) % 2 != 0: right += 1
            return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Determine the maximum dimensions
        max_height = max(img.shape[0] for img in showPool)
        max_width = max(img.shape[1] for img in showPool)

        # Add padding to all images to make them the same size
        showPool = [add_padding(img, (max_height, max_width)) for img in showPool]
        showPool = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img for img in showPool]

        cv2.imshow("Debug", np.hstack(showPool))
        key = cv2.waitKey(0)

        # If the 'ESC' key is pressed, close all debug windows
        if key == 27:
            cv2.destroyAllWindows()

    return name if similarities[name] >= minSSIM else None

if __name__ == "__main__":
    # Test function
    image = cv2.imread("distorted_symbols/Circle (Red Line).jpeg")
    if image is None:
        print("Failed to load image")
    else:
        mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (100, 40, 0), (160, 255, 255))

    source_file = "symbols"
    images = {}
    for image_file_path in os.listdir(source_file):
        image = cv2.imread(os.path.join(source_file, image_file_path))
        if image is not None:
            images[image_file_path] = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (100, 40, 0), (160, 255, 255))

    name = symbol_recognition(mask, images, 0.75, True)
    print("Most Similar Image:", name)