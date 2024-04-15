import cv2
import numpy as np
import urllib.request
import base64

# Function to apply color range masking
def apply_color_mask(image, low_range, high_range):
    lower_bound = np.array(low_range, dtype=np.uint8)
    upper_bound = np.array(high_range, dtype=np.uint8)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def detect_nuclei(image, cell_contours, nucleus_contour_threshold=200):
    nuclei_counts = []
    nuclei_image = image.copy()  # Create a copy of the original image to draw nuclei contours
    for cell_contour in cell_contours:
        # Mask the cell contour
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [cell_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(image, mask)

        # Convert to grayscale and apply thresholding
        gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_masked_image, 1, 255, cv2.THRESH_BINARY)

        # Find contours of nuclei within the cell contour
        nuclei_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count nuclei and store the count
        nuclei_count = len([contour for contour in nuclei_contours if cv2.contourArea(contour) > nucleus_contour_threshold])
        nuclei_counts.append(nuclei_count)

        # Draw nuclei contours on the nuclei image
        cv2.drawContours(nuclei_image, nuclei_contours, -1, (0, 0, 255), 1)

    return nuclei_counts, nuclei_image


# Function to process the image and detect increased number of nucleoli in a cell
def detect_nucleoli(image_bytes):
    try:
        # Decode the image bytes
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply grayscale dilation to enhance boundaries
        kernel = np.ones((5, 5), np.uint8)
        dilated_image = cv2.dilate(gray_image, kernel, iterations=1)

        # Apply adaptive thresholding to create a binary mask of cell boundaries
        _, binary_mask = cv2.threshold(dilated_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours of the cell boundaries
        cell_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect nuclei within each cell
        nuclei_counts, nuclei_image = detect_nuclei(image, cell_contours)

        # Overlay nuclei contours on the original image
        image_with_nuclei_contours = image.copy()
        nuclei_contours = []
        for contour in cell_contours:
            # Mask the cell contour
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            masked_image = cv2.bitwise_and(image, mask)

            # Convert to grayscale and apply thresholding
            gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(gray_masked_image, 1, 255, cv2.THRESH_BINARY)

            # Find contours of nuclei within the cell contour
            nuclei_contours_cell, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Extend the list of nuclei contours
            nuclei_contours.extend(nuclei_contours_cell)

        # Draw nuclei contours
        cv2.drawContours(image_with_nuclei_contours, nuclei_contours, -1, (0, 0, 255), 1)

        # Convert images to base64 strings
        _, original_image_encoded = cv2.imencode('.jpg', image)
        original_image_base64 = base64.b64encode(original_image_encoded).decode('utf-8')

        _, result_image_encoded = cv2.imencode('.jpg', image_with_nuclei_contours)
        result_image_base64 = base64.b64encode(result_image_encoded).decode('utf-8')

        # Convert the nuclei image to base64 string
        _, nuclei_image_encoded = cv2.imencode('.jpg', nuclei_image)
        nuclei_image_base64 = base64.b64encode(nuclei_image_encoded).decode('utf-8')

        # Return the result
        print(nuclei_counts)
        print(len(cell_contours))
        more = sum(count > 1 for count in nuclei_counts)
        print(more)
        return {
            "original_image": original_image_base64,
            "result_image": result_image_base64,
            "nuclei_counts": nuclei_counts,
            "total_cells": len(cell_contours),
            "cells_with_multiple_nuclei": more,
            "nuclei_image": nuclei_image_base64
        }

    except Exception as e:
        print("An error occurred:", str(e))

