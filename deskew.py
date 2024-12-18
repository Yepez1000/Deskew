import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode

def detect_barcode(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # convert to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Gradient-based edge detection
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)  # Horizontal gradient
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)  # Vertical gradient
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)

    # Step 2: Blur and threshold
    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # Step 3: Morphological operations to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 4: Find contours
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour (barcode)
        largest_contour = max(contours, key=cv2.contourArea)

        # Step 5: Get the rotated bounding box
        rect = cv2.minAreaRect(largest_contour)  # Rotated rectangle
        angle = rect[-1]  # Skew angle

        # Adjust the angle for proper rotation
        if angle < -45:
            angle += 90

        print(f"Detected skew angle: {angle:.2f} degrees")

        # Step 6: Rotate the entire image to deskew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Step 7: Find and draw the barcode bounding box on the deskewed image
        # Transform the bounding box points to the rotated image
        box = cv2.boxPoints(rect)  # Get 4 corner points of the rotated rectangle
        box = np.int32(cv2.transform(np.array([box]), rotation_matrix[:2]))[0]  # Apply rotation matrix

        # Draw the bounding box
        print(f"Box coordinates: {box}")

           # Compute box dimensions
        box_height = max(box[:, 1]) - min(box[:, 1])
        box_width = max(box[:, 0]) - min(box[:, 0])

        # If the box height is greater than the width, rotate the entire image 90 degrees
        if box_height > box_width:
            print("Rotating the image by 90 degrees to align the barcode.")
            deskewed_image = cv2.rotate(deskewed_image, cv2.ROTATE_90_CLOCKWISE)


         # Recalculate the box coordinates after 90-degree rotation
            rotated_h, rotated_w = deskewed_image.shape[:2]
            box = np.array([
                [rotated_w - point[1], point[0]] for point in box
            ])


        # Draw the final bounding box
        cv2.drawContours(deskewed_image, [box], -1, (0, 255, 0), 3)

        
            

        plt.imshow(deskewed_image)
        plt.show()
        # Save the output
        cv2.imwrite(output_path, image)
        print(f"Barcode detected and boxed. Output saved to {output_path}")
    else:
        print("No barcode detected.")

def main():
    path = "imagestodeskew"
    for index, image in enumerate(os.listdir(path)):
        image_path = os.path.join(path, image)

        print(image_path)

        if not os.path.isfile(image_path):
            print("not a file")
            continue

        image = cv2.imread(image_path)
        
        if image is None:
            print("image is none")
            continue    

        
        plt.imshow(image)
        plt.show()


        barcode_path = f"barcodetedect/barcode{index}.jpg"

        os.makedirs("barcodetedect", exist_ok=True)


        detect_barcode(image_path, barcode_path)


        # # Define the base paths
        # gpt_path = "/Users/edgaryepez/Developer/AmericanShip/dekewing/gpt_deskew"

     
        # os.makedirs(gpt_path, exist_ok=True)

        # # Save label deskewed image
      
        # deskewed_image, skew_angle = deskew_based_on_barcode(image)
        # # cv2.imwrite(f"{gpt_path}/label_deskewed{index}.jpg", deskewed_image)

main()
