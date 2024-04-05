# from flask import Flask, jsonify, request
# import numpy as np
# import cv2
# import base64
# import json

# app = Flask(__name__)

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)

# def apply_color_mask(image, low_range, high_range):
#     lower_bound = np.array(low_range, dtype=np.uint8)
#     upper_bound = np.array(high_range, dtype=np.uint8)
#     mask = cv2.inRange(image, lower_bound, upper_bound)
#     masked_image = cv2.bitwise_and(image, image, mask=mask)
#     return masked_image

# def remove_light_regions(image):
#     clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
#     enhanced_image = clahe.apply(image)
#     return enhanced_image

# def find_draw_nuclei_boundaries_and_get_sizes(image, min_area=50):
#     kernel = np.ones((5, 5), np.uint8)
#     dilated_image = cv2.dilate(image, kernel, iterations=1)

#     _, binary_mask = cv2.threshold(dilated_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

#     nuclei_count = 0
#     nuclei_sizes = []

#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area >= min_area:
#             cv2.drawContours(result_image, [contour], -1, (0, 0, 255), 1)
#             nuclei_count += 1
#             nuclei_sizes.append(area)

#     nuclei_sizes_array = np.array(nuclei_sizes)

#     return result_image, nuclei_count, nuclei_sizes_array, contours

# def calculate_average_nucleus_size(image_height, nuclei_contours, nuclei_sizes):
#     section_height = image_height // 3
#     top_section_sizes = []
#     middle_section_sizes = []
#     bottom_section_sizes = []

#     for contour, size in zip(nuclei_contours, nuclei_sizes):
#         M = cv2.moments(contour)
#         if M["m00"] != 0:
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])
#         else:
#             cx, cy = 0, 0

#         if 0 <= cy < section_height:
#             top_section_sizes.append(size)
#         elif section_height <= cy < 2 * section_height:
#             middle_section_sizes.append(size)
#         elif 2 * section_height <= cy < image_height:
#             bottom_section_sizes.append(size)

#     average_top_section_size = np.mean(top_section_sizes) if top_section_sizes else 0
#     average_middle_section_size = np.mean(middle_section_sizes) if middle_section_sizes else 0
#     average_bottom_section_size = np.mean(bottom_section_sizes) if bottom_section_sizes else 0

#     return average_top_section_size, average_middle_section_size, average_bottom_section_size

# def draw_horizontal_lines(image, section_height):
#     line_color = (0, 255, 0)
#     line_thickness = 2

#     cv2.line(image, (0, section_height), (image.shape[1], section_height), line_color, line_thickness)
#     cv2.line(image, (0, 2 * section_height), (image.shape[1], 2 * section_height), line_color, line_thickness)

# def increased_nuclei(image_bytes):
#     original_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

#     # Adjusted lower and upper bounds for nucleus detection
#     nucleus_low_range = (90, 55, 50)
#     nucleus_high_range = (255, 255, 255)

#     # Apply the color range masking with the adjusted range for nucleus detection
#     masked_image = apply_color_mask(original_image, nucleus_low_range, nucleus_high_range)

#     # Convert the masked image to grayscale
#     gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

#     # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
#     clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
#     enhanced_image = clahe.apply(gray_masked_image)

#     # Apply negative transformation to make the background black and cells white
#     negative_image = 255 - masked_image

#     # Convert the negative image to grayscale
#     gray_negative_image = cv2.cvtColor(negative_image, cv2.COLOR_BGR2GRAY)

#     # Threshold the negative image to create a binary mask of the white spots
#     _, white_spot_mask = cv2.threshold(gray_negative_image, 200, 255, cv2.THRESH_BINARY)

#     # Find contours of the white spots on the binary mask
#     white_spot_contours, _ = cv2.findContours(white_spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Create a copy of the original masked image to overlay white spot contours
#     image_with_white_spot_contours = masked_image.copy()

#     # Create a list to store the contours of nuclei that meet the area threshold
#     nucleus_contours_above_threshold = []

#     # Iterate through contours and overlay them on the original image
#     for white_spot_contour in white_spot_contours:
#         # Calculate the area of the current white spot contour
#         contour_area = cv2.contourArea(white_spot_contour)

#         # Check if the contour area is above the threshold
#         if contour_area > 10:
#             # Draw contours for white spots that meet the threshold in blue
#             cv2.drawContours(image_with_white_spot_contours, [white_spot_contour], -1, (0, 0, 255), 1)  # Blue color for white spot contours

#             # Add the contour to the list if it meets the area threshold
#             if contour_area > 10:  # Reject nuclei with an area less than 10
#                 nucleus_contours_above_threshold.append(white_spot_contour)

#     # Apply morphological operations to remove noise from the binary mask
#     kernel = np.ones((1, 1), np.uint8)
#     clean_nucleus_mask = cv2.morphologyEx(white_spot_mask, cv2.MORPH_OPEN, kernel)

#     # Find contours of the clean nuclei mask
#     clean_nucleus_contours, _ = cv2.findContours(clean_nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Create a copy of the original masked image to overlay clean nuclei contours
#     image_with_clean_nucleus_contours = negative_image.copy()

#     # Draw contours around the clean nuclei in blue
#     cv2.drawContours(image_with_clean_nucleus_contours, clean_nucleus_contours, -1, (0, 0, 255), 1)

#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

#     # Apply grayscale dilation to enhance boundaries
#     kernel = np.ones((5, 5), np.uint8)
#     dilated_image = cv2.dilate(gray_image, kernel, iterations=2)

#     # Apply adaptive thresholding to create a binary mask of dark spots
#     _, binary_mask = cv2.threshold(dilated_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Find contours of the boundaries
#     image_height = original_image.shape[0]

#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Create an RGB version of the image for visualization
#     result_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#     # Dictionary to store nucleus count per cell
#     nucleus_count_per_cell = {}

#     for index, clean_nucleus_contour in enumerate(clean_nucleus_contours):
#         # Create a mask for the current nucleus contour
#         cell_mask = np.zeros(gray_image.shape, dtype=np.uint8)
#         cv2.drawContours(cell_mask, [clean_nucleus_contour], -1, 255, thickness=cv2.FILLED)

#         # Bitwise AND operation to extract the nucleus from the masked image
#         nucleus = cv2.bitwise_and(gray_image, gray_image, mask=cell_mask)

#         # Count the number of nuclei in the current cell
#         _, nuclei_contours, _ = cv2.findContours(nucleus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         nucleus_count_per_cell[index] = len(nuclei_contours)

#     # Count cells with more than one nucleus
#     cells_with_multiple_nuclei = sum(1 for count in nucleus_count_per_cell.values() if count > 1)

#     # Encode images to base64 for JSON response
#     _, img_encoded_result = cv2.imencode('.jpg', image_with_clean_nucleus_contours)
#     img_base64_result = base64.b64encode(img_encoded_result).decode('utf-8')

#     _, img_encoded_original = cv2.imencode('.jpg', original_image)
#     img_base64_original = base64.b64encode(img_encoded_original).decode('utf-8')

#     return {
#         'totalNuclei': len(clean_nucleus_contours),
#         'resultImage': img_base64_result,
#         'originalImage': img_base64_original,
#         'nucleusCountPerCell': nucleus_count_per_cell,
#         'cellsWithMultipleNuclei': cells_with_multiple_nuclei
#     }




# if __name__ == '_main_':
#     app.run(host='localhost', port=5000, debug=True)


from flask import Flask, jsonify, request
import numpy as np
import cv2
import base64
import json

app = Flask(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def apply_color_mask(image, low_range, high_range):
    lower_bound = np.array(low_range, dtype=np.uint8)
    upper_bound = np.array(high_range, dtype=np.uint8)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def remove_light_regions(image):
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def find_draw_nuclei_boundaries_and_get_sizes(image, min_area=50):
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)

    _, binary_mask = cv2.threshold(dilated_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    nuclei_count = 0
    nuclei_sizes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(result_image, [contour], -1, (0, 0, 255), 1)
            nuclei_count += 1
            nuclei_sizes.append(area)

    nuclei_sizes_array = np.array(nuclei_sizes)

    return result_image, nuclei_count, nuclei_sizes_array, contours

def calculate_average_nucleus_size(image_height, nuclei_contours, nuclei_sizes):
    section_height = image_height // 3
    top_section_sizes = []
    middle_section_sizes = []
    bottom_section_sizes = []

    for contour, size in zip(nuclei_contours, nuclei_sizes):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        if 0 <= cy < section_height:
            top_section_sizes.append(size)
        elif section_height <= cy < 2 * section_height:
            middle_section_sizes.append(size)
        elif 2 * section_height <= cy < image_height:
            bottom_section_sizes.append(size)

    average_top_section_size = np.mean(top_section_sizes) if top_section_sizes else 0
    average_middle_section_size = np.mean(middle_section_sizes) if middle_section_sizes else 0
    average_bottom_section_size = np.mean(bottom_section_sizes) if bottom_section_sizes else 0

    return average_top_section_size, average_middle_section_size, average_bottom_section_size

def draw_horizontal_lines(image, section_height):
    line_color = (0, 255, 0)
    line_thickness = 2

    cv2.line(image, (0, section_height), (image.shape[1], section_height), line_color, line_thickness)
    cv2.line(image, (0, 2 * section_height), (image.shape[1], 2 * section_height), line_color, line_thickness)

def increased_nuclei(image_bytes):
    original_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Adjusted lower and upper bounds for nucleus detection
    nucleus_low_range = (90, 55, 50)
    nucleus_high_range = (255, 255, 255)

    # Apply the color range masking with the adjusted range for nucleus detection
    masked_image = apply_color_mask(original_image, nucleus_low_range, nucleus_high_range)


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

    # Create a copy of the original masked image to overlay white spot contours
    image_with_white_spot_contours = masked_image.copy()

    # Create a list to store the contours of nuclei that meet the area threshold
    nucleus_contours_above_threshold = []

    # Iterate through contours and overlay them on the original image
    for white_spot_contour in white_spot_contours:
        # Calculate the area of the current white spot contour
        contour_area = cv2.contourArea(white_spot_contour)

        # Check if the contour area is above the threshold
        if contour_area > 10:
            # Draw contours for white spots that meet the threshold in blue
            cv2.drawContours(image_with_white_spot_contours, [white_spot_contour], -1, (0, 0, 255), 1)  # Blue color for white spot contours

            # Add the contour to the list if it meets the area threshold
            if contour_area > 10:  # Reject nuclei with an area less than 10
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
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply grayscale dilation to enhance boundaries
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(gray_image, kernel, iterations=2)

    # Apply adaptive thresholding to create a binary mask of dark spots
    _, binary_mask = cv2.threshold(dilated_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the boundaries
    image_height = original_image.shape[0]

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an RGB version of the image for visualization
    result_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Dictionary to store nucleus count per cell
    nucleus_count_per_cell = {}

    for index, clean_nucleus_contour in enumerate(clean_nucleus_contours):
        # Create a mask for the current nucleus contour
        cell_mask = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.drawContours(cell_mask, [clean_nucleus_contour], -1, 255, thickness=cv2.FILLED)

        # Bitwise AND operation to extract the nucleus from the masked image
        nucleus = cv2.bitwise_and(gray_image, gray_image, mask=cell_mask)

        # Count the number of nuclei in the current cell
        nuclei_contours, _ = cv2.findContours(nucleus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nucleus_count_per_cell[index] = len(nuclei_contours)

    # Count cells with more than one nucleus
    cells_with_multiple_nuclei = sum(1 for count in nucleus_count_per_cell.values() if count > 1)

    # Encode images to base64 for JSON response
    _, img_encoded_result = cv2.imencode('.jpg', image_with_clean_nucleus_contours)
    img_base64_result = base64.b64encode(img_encoded_result).decode('utf-8')

    _, img_encoded_original = cv2.imencode('.jpg', original_image)
    img_base64_original = base64.b64encode(img_encoded_original).decode('utf-8')

    return {
        'totalNuclei': len(clean_nucleus_contours),
        'resultImage': img_base64_result,
        'originalImage': img_base64_original,
        'nucleusCountPerCell': nucleus_count_per_cell,
        'cellsWithMultipleNuclei': cells_with_multiple_nuclei
    }

if __name__ == 'main':
    app.run(host='localhost', port=5000, debug=True)