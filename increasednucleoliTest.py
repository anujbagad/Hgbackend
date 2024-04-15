import cv2
import numpy as np
from matplotlib import pyplot as plt
import base64

# Function to apply color range masking
def apply_color_mask(image, low_range, high_range):
    lower_bound = np.array(low_range, dtype=np.uint8)
    upper_bound = np.array(high_range, dtype=np.uint8)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

# Function to apply connected component labeling
def label_connected_components(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    return num_labels, labels, stats, centroids

# Function to remove light regions from a grayscale image using CLAHE
def remove_light_regions(image):
    # Apply CLAHE to enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(result_image)
    return enhanced_image

# Function to detect and count black spots (objects) in an image
def count_black_spots(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary mask of the black spots
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the black spots
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on a copy of the original image for visualization
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB for color drawing
    cv2.drawContours(image_with_contours, contours, -1, (255, 0, 0), 1)  # Red color for contours

    # Count the detected black spots
    spot_count = len(contours)

    # Return the count and the image with contours
    return spot_count, image_with_contours

# Define the file path of the uploaded image
# image_path = "uploads/4.jpg"  # Update this with your image file path

# Load the image from the file path
def modelOutput(image_path):
    try:
        image = cv2.imread(image_path)

        # Adjusted lower and upper bounds for cell detection
        cell_low_range = (52, 52, 52)
        cell_high_range = (255, 255, 255)

        # Apply the color range masking with the adjusted range for cell detection
        masked_image = apply_color_mask(image, cell_low_range, cell_high_range)

        # Convert the masked image to grayscale
        gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray_masked_image)

        # Apply negative transformation to make the background black and cells white
        negative_image = 255 - masked_image

        # Convert the negative image to grayscale
        gray_negative_image = cv2.cvtColor(negative_image, cv2.COLOR_BGR2GRAY)

        # Threshold the negative image to create a binary mask of the white spots
        _, white_spot_mask = cv2.threshold(gray_negative_image, 200, 255, cv2.THRESH_BINARY)

        # Find contours of the white spots on the binary mask
        white_spot_contours, _ = cv2.findContours(white_spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set the area threshold for detecting white spots (you can adjust this value)
        min_nucleus_area = 15  # Change this value as needed

        # Create a copy of the original masked image to overlay white spot contours
        image_with_white_spot_contours = masked_image.copy()

        # Create a list to store the contours of nuclei that meet the area threshold
        nucleus_contours_above_threshold = []

        # Iterate through contours and overlay them on the original image
        for white_spot_contour in white_spot_contours:
            # Calculate the area of the current white spot contour
            contour_area = cv2.contourArea(white_spot_contour)

            # Check if the contour area is above the threshold
            if contour_area > min_nucleus_area:
                # Draw contours for white spots that meet the threshold in blue
                cv2.drawContours(image_with_white_spot_contours, [white_spot_contour], -1, (0, 0, 255), 1)  # Blue color for white spot contours

                # Add the contour to the list
                nucleus_contours_above_threshold.append(white_spot_contour)

        # Apply morphological operations to remove noise from the binary mask
        kernel = np.ones((1, 1), np.uint8)
        clean_nucleus_mask = cv2.morphologyEx(white_spot_mask, cv2.MORPH_OPEN, kernel)

        # Find contours of the clean nuclei mask
        clean_nucleus_contours, _ = cv2.findContours(clean_nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original masked image to overlay clean nuclei contours
        image_with_clean_nucleus_contours = negative_image.copy()

        # Draw contours around the clean nuclei in blue
        cv2.drawContours(image_with_clean_nucleus_contours, clean_nucleus_contours, -1, (0, 0, 255), 1)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply grayscale dilation to enhance boundaries
        kernel = np.ones((5, 5), np.uint8)
        dilated_image = cv2.dilate(gray_image, kernel, iterations=1)

        # Apply adaptive thresholding to create a binary mask of dark spots
        _, binary_mask = cv2.threshold(dilated_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours of the boundaries
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an RGB version of the image for visualization
        result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize count for boundaries
        boundary_count = 0

        # List to store the number of nuclei detected within each cell boundary
        nuclei_counts = []

        # print("contours :", contours)

        # Iterate through contours and draw boundaries based on area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= 150:  # You can adjust the area threshold as needed
                cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 1)
                boundary_count += 1

                # Count the number of nuclei within this cell boundary
                nuclei_within_boundary = 0
                for nucleus_contour in nucleus_contours_above_threshold:
                    # Convert the point to integers
                    nucleus_contour_point = tuple(map(int, nucleus_contour[0][0]))

                    # Check if the nucleus contour is contained within the cell boundary contour
                    if cv2.pointPolygonTest(contour, nucleus_contour_point, False) > 0:
                        nuclei_within_boundary += 1

                if nuclei_within_boundary > 1 and nuclei_within_boundary < 6:  # Check if there are more than 1 nuclei
                    nuclei_counts.append(nuclei_within_boundary)

        # Overlay nuclei contours on the result image in blue
        # Overlay nuclei contours on the result image in blue
        cv2.drawContours(result_image, nucleus_contours_above_threshold, -1, (0, 0, 255), thickness=cv2.FILLED)


        # Calculate the area of each individual nucleus contour
        nuclei_areas = [cv2.contourArea(contour) for contour in nucleus_contours_above_threshold]

        # Calculate the average area of nuclei contours
        if len(nuclei_areas) > 0:
            average_nuclei_area = sum(nuclei_areas) / len(nuclei_areas)
        else:
            average_nuclei_area = 0

        # Divide the image into three equal sections horizontally
        image_height, image_width, _ = image.shape
        section_height = image_height // 3

        # Calculate the y-coordinates for the three sections
        top_section_start = 0
        middle_section_start = section_height
        bottom_section_start = 2 * section_height

        # Initialize severity level
        severity = None

        # Iterate through contours and draw boundaries based on area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= 150:  # You can adjust the area threshold as needed
                # Check which section the cell belongs to
                center_y = int((contour[0][0][1] + contour[-1][0][1]) / 2)

                if top_section_start <= center_y < middle_section_start:
                    severity = "Severe"
                elif middle_section_start <= center_y < bottom_section_start:
                    severity = "Moderate"
                elif bottom_section_start <= center_y < image_height:
                    severity = "Mild"

        # Print the total number of boundaries detected
        print("Total Cell Boundaries detected:", boundary_count)
        print("Nuclei Count:", len(clean_nucleus_contours))

        # Print the number of nuclei detected within each cell boundary above 1
        print("Nuclei Counts within Cell Boundaries (above 1):", nuclei_counts)

        # Print the average area of nuclei contours
        print("Average Nuclei Area:", average_nuclei_area)

        # Print the individual areas of nuclei contours
        print("Individual Nuclei Areas:", nuclei_areas)
        print("Class:", severity)

        # Convert images to base64 strings
        _, original_image_encoded = cv2.imencode('.jpg', image)
        original_image_base64 = base64.b64encode(original_image_encoded).decode('utf-8')

        # Convert the result image to base64 string
        _, result_image_encoded = cv2.imencode('.jpg', result_image)
        result_image_base64 = base64.b64encode(result_image_encoded).decode('utf-8')

        return {
            "original_image": original_image_base64,
            "result_image": result_image_base64,
            "boundary_count":boundary_count,
            "nuclei_counts":len(nuclei_counts),
            "average_nuclei_area":average_nuclei_area,
            "nuclei_areas":len(nuclei_areas),
            "severity":severity
        }

    except Exception as e:
        print("An error occurred:", str(e))
