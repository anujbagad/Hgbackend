import cv2
import numpy as np
import base64

# Function to apply color range masking
def apply_color_mask(image, low_range, high_range):
    lower_bound = np.array(low_range, dtype=np.uint8)
    upper_bound = np.array(high_range, dtype=np.uint8)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

# Function to calculate ratio of each nucleus to cell area within each cell contour
def calculate_nucleus_to_cell_ratio(nucleus_contours, cell_contours):
    nucleus_to_cell_ratios = []
    for cell_idx, cell_contour in enumerate(cell_contours):
        cell_area = cv2.contourArea(cell_contour)
        has_nucleus_inside = False
        for nucleus_contour in nucleus_contours:
            nucleus_area = cv2.contourArea(nucleus_contour)
            if len(cell_contour) > 0 and len(nucleus_contour) > 0:
                # Calculate the centroid of the nucleus
                M = cv2.moments(nucleus_contour)
                if M["m00"] != 0:
                    nucleus_centroid_x = int(M["m10"] / M["m00"])
                    nucleus_centroid_y = int(M["m01"] / M["m00"])
                    # Check if the centroid of the nucleus is inside the cell contour
                    if cv2.pointPolygonTest(cell_contour, (nucleus_centroid_x, nucleus_centroid_y), False) >= 0:
                        has_nucleus_inside = True
                        nucleus_to_cell_ratio = nucleus_area / cell_area
                        nucleus_to_cell_ratios.append(nucleus_to_cell_ratio)
                        break
        # If cell has no nucleus inside, skip it
        if not has_nucleus_inside:
            continue
    return nucleus_to_cell_ratios

# Function to divide image into three equal horizontal sections
def divide_image(image):
    height, width, _ = image.shape
    section_height = height // 3
    top_section = image[:section_height, :, :]
    middle_section = image[section_height:2 * section_height, :, :]
    bottom_section = image[2 * section_height:, :, :]
    return top_section, middle_section, bottom_section



result = {}

def process_ratio(image_bytes):
    try:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        # Adjusted lower and upper bounds for cell detection
        cell_low_range = (52, 52, 52)
        cell_high_range = (255, 255, 255)

        # Apply the color range masking with the adjusted range for cell detection
        masked_image = apply_color_mask(image, cell_low_range, cell_high_range)

        # Convert the masked image to grayscale
        gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Apply negative transformation to make the background black and cells white
        negative_image = 255 - masked_image

        # Convert the negative image to grayscale
        gray_negative_image = cv2.cvtColor(negative_image, cv2.COLOR_BGR2GRAY)

        # Threshold the negative image to create a binary mask of the white spots
        _, white_spot_mask = cv2.threshold(gray_negative_image, 155, 255, cv2.THRESH_BINARY)  # Adjust threshold value here

        # Find contours of the white spots on the binary mask
        white_spot_contours, _ = cv2.findContours(white_spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set the area threshold for detecting white spots (you can adjust this value)
        min_nucleus_area = 30  # Change this value as needed

        # Create a list to store the contours of nuclei that meet the area threshold
        nucleus_contours_above_threshold = []

        # Iterate through contours and overlay them on the original image
        for white_spot_contour in white_spot_contours:
            # Calculate the area of the current white spot contour
            contour_area = cv2.contourArea(white_spot_contour)

            # Check if the contour area is above the threshold
            if contour_area > min_nucleus_area:
                # Add the contour to the list
                nucleus_contours_above_threshold.append(white_spot_contour)

        # Find boundaries, draw them, and get contour sizes for cells
        def find_and_draw_boundaries_with_sizes(image, min_area=200):
            # Apply grayscale dilation to enhance boundaries
            kernel = np.ones((5, 5), np.uint8)
            dilated_image = cv2.dilate(image, kernel, iterations=1)

            # Apply adaptive thresholding to create a binary mask of dark spots
            _, binary_mask = cv2.threshold(dilated_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find contours of the boundaries
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create an RGB version of the image for visualization
            result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Initialize count for boundaries and a list to store contour sizes
            boundary_count = 0
            contour_sizes = []

            # Iterate through contours and draw boundaries based on area
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    cv2.drawContours(result_image, [contour], -1, (255, 0, 0), 1)
                    boundary_count += 1
                    contour_sizes.append(area)

            return result_image, boundary_count, contour_sizes, contours

        # Convert the masked image to grayscale
        gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Remove light regions from the grayscale image using CLAHE
        def remove_light_regions(image):
            # Apply CLAHE to enhance local contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(image)

            return enhanced_image

        # Remove light regions from the grayscale image using CLAHE
        cleaned_image = remove_light_regions(gray_masked_image)

        # Find boundaries, draw them, and get contour sizes for cells
        result_image, boundary_count, contour_sizes, contours = find_and_draw_boundaries_with_sizes(cleaned_image, min_area=250)

        # Combine images
        combined_image = result_image

        # Draw nuclei contours on the combined image
        cv2.drawContours(combined_image, nucleus_contours_above_threshold, -1, (0, 255, 0), 1)

        # Number the contours from 1 for cells
        for idx, _ in enumerate(contour_sizes):
            cv2.putText(combined_image, str(idx + 1), (10, (idx + 1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Minimum and maximum cell size thresholds
        min_cell_area = 200
        max_cell_area = 1800

        # Divide image into three equal horizontal sections
        top_section, middle_section, bottom_section = divide_image(combined_image)
        # Get the heights of each section
        height, width, _ = image.shape
        section_height = height // 3

        # Calculate top, middle, and bottom indices for cell contours
        top_index = int(section_height / 2)
        middle_index = int(section_height / 2) + section_height
        bottom_index = int(section_height / 2) + 2 * section_height

        # Filter cell contours for each section
        top_cell_contours = [contour for contour in contours if cv2.boundingRect(contour)[1] < top_index]
        middle_cell_contours = [contour for contour in contours if top_index <= cv2.boundingRect(contour)[1] < middle_index]
        bottom_cell_contours = [contour for contour in contours if cv2.boundingRect(contour)[1] >= bottom_index]
        # Calculate the average ratio for each section
        top_ratio = np.mean(calculate_nucleus_to_cell_ratio(nucleus_contours_above_threshold, top_cell_contours))
        middle_ratio = np.mean(calculate_nucleus_to_cell_ratio(nucleus_contours_above_threshold, middle_cell_contours))
        bottom_ratio = np.mean(calculate_nucleus_to_cell_ratio(nucleus_contours_above_threshold, bottom_cell_contours))

        # Return the average ratios for each section
        overall_average_ratio = (top_ratio + middle_ratio + bottom_ratio) / 3
        result["top_ratio"] = top_ratio
        result["middle_ratio"] = middle_ratio
        result["bottom_ratio"] = bottom_ratio
        result["overall_average_ratio"] = overall_average_ratio
        print(overall_average_ratio)
        print(top_ratio)
        print(middle_ratio)
        print(bottom_ratio)
        
        classification = ""
        if bottom_ratio > middle_ratio:
            classification = "Mild"
        elif middle_ratio > bottom_ratio:
            classification = "Moderate"
        elif top_ratio > middle_ratio:
            classification = "Severe"

        # Convert images to base64 strings
        _, original_image_encoded = cv2.imencode('.png', image)
        _, result_image_encoded = cv2.imencode('.png', combined_image)
        original_image_base64 = base64.b64encode(original_image_encoded).decode('utf-8')
        result_image_base64 = base64.b64encode(result_image_encoded).decode('utf-8')
        return {
            "top_ratio": top_ratio,
            "middle_ratio": middle_ratio,
            "bottom_ratio": bottom_ratio,
            "overall_average_ratio": overall_average_ratio,
            "original_image": original_image_base64,
            "result_image": result_image_base64,
            "classify" : classification
        }

    except Exception as e:
        result["error"] = str(e)
