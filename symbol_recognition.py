import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

import tkinter as tk
from tkinter import filedialog
from tkinter import colorchooser
import os

def main():
    # Specify the file types to be displayed
    file_types = [("Image files", "*.jpg"), 
                ("Image files", "*.jpeg"), 
                ("Image files", "*.png"), 
                ("Image files", "*.gif"), 
                ("All files", "*.*")]
    
    # Set the initial directory to the script folder
    script_folder = os.path.dirname(os.path.abspath(__file__))

    lowHSV = (140, 100, 100)
    highHSV = (180, 255, 255)

    # Create the Tkinter root
    root = tk.Tk()
    file_path = tk.StringVar()
    folder_path = tk.StringVar()

    # Create a button to select the file
    select_button = tk.Button(root, text="Select Image To Identify", command=lambda: file_path.set(filedialog.askopenfilename(initialdir=script_folder, filetypes=file_types)))
    select_button.pack()

    # Create a button to select multiple files
    select_button = tk.Button(root, text="Select Identification Source", command=lambda: folder_path.set(filedialog.askdirectory(initialdir=script_folder)))
    select_button.pack()

    def rgb_to_hsv(rgb):
        # Convert the RGB values to the range 0-255
        rgb = [int(x) for x in rgb]

        # Convert the RGB color to a NumPy array and reshape it to a 3D array
        color_array = np.uint8([[list(rgb)]])

        # Convert the color to HSV using cv2.cvtColor()
        hsv_color = cv2.cvtColor(color_array, cv2.COLOR_RGB2HSV)

        # Return the HSV color as a tuple
        return tuple(hsv_color[0][0])

    def pick_low_HSV():
        # Open the color picker dialog and get the chosen color
        rgb_color = colorchooser.askcolor()[0]

        # If a color was chosen (i.e., the user didn't cancel the dialog)
        if rgb_color is not None:
            # Convert the RGB color to HSV
            lowHSV = rgb_to_hsv(rgb_color)
            print(f"HSV color: {lowHSV}")

    # Create a button to open the color picker for the low HSV values
    color_button = tk.Button(root, text="Pick Low HSV", command=pick_low_HSV)
    color_button.pack()

    def pick_high_HSV():
        # Open the color picker dialog and get the chosen color
        rgb_color = colorchooser.askcolor()[0]

        # If a color was chosen (i.e., the user didn't cancel the dialog)
        if rgb_color is not None:
            # Convert the RGB color to HSV
            highHSV = rgb_to_hsv(rgb_color)
            print(f"HSV color: {highHSV}")

    # Create a button to open the color picker for the high HSV values
    color_button = tk.Button(root, text="Pick High HSV", command=pick_high_HSV)
    color_button.pack()

    def process_button_function():
        nonlocal file_path
        nonlocal folder_path

        if not file_path.get():
            print("Image Not Selected!")
            tk.messagebox.showerror("Error", "Image Not Selected!")
            return
        
        if not folder_path.get():
            print("Source Not Selected!")
            tk.messagebox.showerror("Error", "Source Not Selected!")
            return
        
        image = cv2.imread(file_path.get())
        if image is None:
            print("Error Reading Image!")
            tk.messagebox.showerror("Error", "Error Reading Image!")
            return
        
        image_file_paths = os.listdir(folder_path.get())
        images = []
        images_skipped = []
        
        for image_file_path in image_file_paths:
            image_path = os.path.join(folder_path.get(), image_file_path)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image: {image_path}")
                images_skipped.append(image_path)
            else:
                images.append(image)

        if not images:
            print("No valid source found!")
            tk.messagebox.showerror("Error", "No valid source found!")
            return
        
        if images_skipped:
            tk.messagebox.showwarning("Warning", f"Skipped files:\n{"\n".join(images_skipped)}")

        process_symbol_recognition(image, lowHSV, highHSV, images)

    # Create a button to process the image
    process_button = tk.Button(root, text="Identify Image", command=process_button_function)
    process_button.pack()

    # Start the Tkinter event loop
    root.mainloop()

def process_symbol_recognition(image, lowHSV, highHSV, source):
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

if __name__ == "__main__":
    # main()

    # Skip UI for debug
    # ---
    image = cv2.imread("distorted_symbols/Umbrella (Yellow Line).png")
    source_file = "symbols"

    images = [cv2.imread(os.path.join("symbols", image_file_path)) for image_file_path in os.listdir(source_file) if cv2.imread(os.path.join(source_file, image_file_path)) is not None]
    process_symbol_recognition(image, (140, 100, 100), (180, 255, 255), images)
    # ---