# cellsize.py 
import os
import cv2
import numpy as np
from flask import jsonify, request
import base64
import json
import skimage.io
import requests
from PIL import Image
from io import BytesIO
import histomicstk as htk

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def divide_image_into_sections(image):
    # Divide the image into three equal sections horizontally
    image_height, image_width = image.shape
    section_height = image_height // 3

    # Calculate the y-coordinates for the three sections
    top_section_start = 0
    middle_section_start = section_height
    bottom_section_start = 2 * section_height

    # Extract three sections
    top_section = image[top_section_start:middle_section_start, :]
    middle_section = image[middle_section_start:bottom_section_start, :]
    bottom_section = image[bottom_section_start:, :]

    return top_section, middle_section, bottom_section


def image_to_base64(image):
    # Convert image to PIL format
    pil_image = Image.fromarray(image)
    # Convert PIL image to base64 string
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


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


def classify_cell_size(image_bytes, dataset_path):
    original_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Calculate the average cell size for the input image
    cell_low_range = (52, 52, 52)
    cell_high_range = (255, 255, 255)
    masked_image = apply_color_mask(original_image, cell_low_range, cell_high_range)
    gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    result_image, nuclei_count, nuclei_sizes, nuclei_contours = find_draw_nuclei_boundaries_and_get_sizes(
        gray_masked_image, min_area=15
    )
    image_height, _, _ = original_image.shape
    avg_top_input, avg_mid_input, avg_bottom_input = calculate_average_nucleus_size(
        image_height, nuclei_contours, nuclei_sizes
    )
    total_cell_size = np.sum(nuclei_sizes)

    # Compare input with the dataset averages
    classification_result = {
        'TotalNuclei': nuclei_count,
        'AverageTopInput': avg_top_input,
        'AverageMidInput': avg_mid_input,
        'AverageBottomInput': avg_bottom_input,
        'ResultImage': None,
        'OriginalImage': None,
        'Classification': '',
    }

    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)

        if os.path.isdir(subdir_path):
            subdir_averages = {'Top': [], 'Middle': [], 'Bottom': []}

            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)

                # Read and process each image in the dataset
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, nuclei_count, nuclei_sizes, nuclei_contours = find_draw_nuclei_boundaries_and_get_sizes(
                    gray_image, min_area=20
                )
                image_height, _, _ = image.shape
                avg_top, avg_mid, avg_bottom = calculate_average_nucleus_size(
                    image_height, nuclei_contours, nuclei_sizes
                )

                subdir_averages['Top'].append(avg_top)
                subdir_averages['Middle'].append(avg_mid)
                subdir_averages['Bottom'].append(avg_bottom)

            avg_top_subdir = np.mean(subdir_averages['Top']) if any(subdir_averages['Top']) else 0
            avg_mid_subdir = np.mean(subdir_averages['Middle']) if any(subdir_averages['Middle']) else 0
            avg_bottom_subdir = np.mean(subdir_averages['Bottom']) if any(subdir_averages['Bottom']) else 0

            print(f'{subdir} Averages:')
            print({'Top': avg_top_subdir, 'Mid': avg_mid_subdir, 'Bottom': avg_bottom_subdir})

            if (
                avg_top_input >= 0.8*avg_top_subdir
                and avg_mid_input >= 0.8*avg_mid_subdir
                or avg_bottom_input >= 0.8*avg_bottom_subdir
            ):
                classification_result['Classification'] = subdir
                print({subdir})
                break
            elif (
                avg_top_input >= 0.8*avg_top_subdir
                and avg_bottom_input >= 0.8*avg_bottom_subdir
                or avg_mid_input >= 0.8*avg_mid_subdir
            ):
                classification_result['Classification'] = subdir
                print({subdir})
                break
            elif (
                avg_bottom_input >= 0.8*avg_bottom_subdir
                and avg_mid_input >= 0.8*avg_mid_subdir
                or avg_top_input >= 0.8*avg_top_subdir
            ):
                classification_result['Classification'] = subdir
                print({subdir})
                break
            else:
                classification_result['Classification'] = 'Normal'


    draw_horizontal_lines(result_image, image_height // 3)
    _, img_encoded_result = cv2.imencode('.jpg', result_image)
    img_base64_result = base64.b64encode(img_encoded_result).decode('utf-8')

    _, img_encoded_original = cv2.imencode('.jpg', original_image)
    img_base64_original = base64.b64encode(img_encoded_original).decode('utf-8')

    avg = avg_top + avg_mid + avg_bottom / 3

    print('Total Nuclei:', classification_result.get('TotalNuclei'))
    print('Avg cell size (Top): ', classification_result.get('AverageTopInput'))
    print('Avg cell size (Mid): ', classification_result.get('AverageMidInput'))
    print('Avg cell size (Bottom): ', classification_result.get('AverageBottomInput'))
    print('Classification:', classification_result.get('Classification'))
    print('AVERAGE:', avg)

    return {
        'totalNuclei': classification_result.get('TotalNuclei'),
        'totalCellSize': total_cell_size,
        'averageTop': avg_top_input,
        'averageMiddle': avg_mid_input,
        'averageBottom': avg_bottom_input,
        'resultImage': img_base64_result,
        'originalImage': img_base64_original,
        'classificationResult': classification_result.get('Classification'),
    }

def process_nucleus_image(image_bytes):
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
            nucleus_contours_above_threshold.append(white_spot_contour)# Blue color for white spot contours

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

    average_top, average_middle, average_bottom = calculate_average_nucleus_size(
        image_height, nucleus_contours_above_threshold, [cv2.contourArea(contour) for contour in nucleus_contours_above_threshold]
    )

    # Print the total number of boundaries detected
    print("Nuclei Count:", len(clean_nucleus_contours))
    print(average_top)
    print(average_middle)
    print(average_bottom)

    # Encode images to base64 for JSON response
    _, img_encoded_result = cv2.imencode('.jpg', image_with_clean_nucleus_contours)
    img_base64_result = base64.b64encode(img_encoded_result).decode('utf-8')

    _, img_encoded_original = cv2.imencode('.jpg', original_image)
    img_base64_original = base64.b64encode(img_encoded_original).decode('utf-8')

    return {
        'totalNuclei': len(clean_nucleus_contours),
        'resultImage': img_base64_result,
        'originalImage': img_base64_original,
        'averageTop': average_top,
        'averageMiddle': average_middle,
        'averageBottom': average_bottom,
    }

def detect_hyperchromasia(image_bytes):
    try:
        # Read the original image
        original_image = Image.open(BytesIO(image_bytes))

        # Convert image to numpy array
        img = np.array(original_image)

        # create stain to color map
        stainColorMap = {
            'hematoxylin': [0.65, 0.70, 0.29],
            'eosin': [0.02, 0.99, 0.11],
            'dab': [0.27, 0.57, 0.78],
            'null': [0.0, 0.0, 0.0]
        }

        # specify stains of the input image
        stain_1 = 'hematoxylin'   # nuclei stain
        stain_2 = 'eosin'         # cytoplasm stain
        stain_3 = 'null'          # set to null if the input contains only two stains

        # create stain matrix
        W = np.array([stainColorMap[stain_1],
                      stainColorMap[stain_2],
                      stainColorMap[stain_3]]).T

        # perform standard color deconvolution
        im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(img, W).Stains

        # Convert result image to base64 string
        result_image_base64 = image_to_base64(im_stains[:, :, :3])

        # Calculate the intensity of each nucleus
        nucleus_intensity = im_stains[:, :, 0]

        # Divide the image into three sections horizontally and calculate average intensity for each section
        top_section, middle_section, bottom_section = divide_image_into_sections(nucleus_intensity)

        avg_intensity_top_section = np.mean(top_section)
        avg_intensity_middle_section = np.mean(middle_section)
        avg_intensity_bottom_section = np.mean(bottom_section)

        # Calculate overall average intensity
        overall_avg_intensity = (avg_intensity_top_section + avg_intensity_middle_section + avg_intensity_bottom_section) / 3

        # Determine the class based on intensity values
        if avg_intensity_bottom_section > avg_intensity_top_section:
            classification = "Severe"
        elif avg_intensity_bottom_section < avg_intensity_middle_section:
            classification = "Moderate"
        elif avg_intensity_bottom_section > avg_intensity_middle_section:
            classification = "Mild"
        else:
            classification = "Normal"

        # Convert original image to base64 string
        original_image_base64 = image_to_base64(img)

        return {
            "original_image": original_image_base64,
            "result_image": result_image_base64,
            "overall_average_intensity": overall_avg_intensity,
            "classification": classification,
            "average_intensity_top_section": avg_intensity_top_section,
            "average_intensity_middle_section": avg_intensity_middle_section,
            "average_intensity_bottom_section": avg_intensity_bottom_section
        }

    except Exception as e:
        return {'error': str(e)}

# Function to read an image from a file path
def read_image_from_path(path):
    img = cv2.imread(path)
    return img

# Function to find a matching template in the main image
def find_matching_template(main_image, template_image, threshold):
    main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    result = cv2.matchTemplate(main_image_gray, template_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        top_left = max_loc
        h, w = template_image.shape[:2]
        bottom_right = (top_left[0] + w, top_left[1] + h)

        matched_img = main_image.copy()
        cv2.rectangle(matched_img, top_left, bottom_right, (0, 255, 0), 2)

        return matched_img, max_val, top_left, bottom_right
    else:
        return None, 0, (0, 0), (0, 0)

# Function to get the position of the contour
def get_contour_position(main_image_height, top_left):
    image_height_third = main_image_height // 3
    contour_center_y = top_left[1] + (top_left[1] // 2)  # Y-coordinate of the center of the contour

    if contour_center_y <= image_height_third:
        return "Severe"
    elif contour_center_y <= 2 * image_height_third:
        return "Moderate"
    else:
        return "Mild"

def detect_keratin_figures(image_bytes, template_folder_path):
    try:
        # Read the original image
        original_image = Image.open(BytesIO(image_bytes))

        # Convert image to numpy array
        img = np.array(original_image)
        threshold = 0.4
        best_matching_score = 0
        best_top_left = (0, 0)
        best_bottom_right = (0, 0)
        best_grade = "Normal"

        match_found = False
        for template_file in os.listdir(template_folder_path):
            template_path = os.path.join(template_folder_path, template_file)

            template_image = read_image_from_path(template_path)

            if template_image is not None:
                if len(template_image.shape) == 3:
                    template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

                matched_img, matching_score, top_left, bottom_right = find_matching_template(img, template_image, threshold)

                if matching_score > best_matching_score:
                    best_matching_score = matching_score
                    best_top_left = top_left
                    best_bottom_right = bottom_right
                    main_image_height = img.shape[0]
                    best_grade = get_contour_position(main_image_height, top_left)

                if matched_img is not None:
                    contour_img = img.copy()
                    cv2.rectangle(contour_img, top_left, bottom_right, (0, 255, 0), 2)

                    match_found = True

        if not match_found:
            return {
                "original_image": None,
                "result_image": None,
                "Keratin_Pearls_score": 0,
                "Keratin_Pearls_grade": "Normal",
                "Keratin_Pearls_found": "No presence of Keratin Pearls observed"
            }
        else:
            contour_img = img.copy()
            cv2.rectangle(contour_img, best_top_left, best_bottom_right, (0, 255, 0), 2)

            # Convert result image to base64 string
            result_image_base64 = image_to_base64(contour_img)
            og_image_base64 = image_to_base64(img)

            # Save the result image to a BytesIO buffer
            result_image_buffer = BytesIO()
            Image.fromarray(contour_img).save(result_image_buffer, format='JPEG')
            result_image_bytes = result_image_buffer.getvalue()
            result_image_base64 = image_to_base64(contour_img)

            return {
                "original_image": og_image_base64,
                "result_image": result_image_base64 if best_matching_score > 0 else None,
                "Keratin_Pearls_score": best_matching_score,
                "Keratin_Pearls_grade": best_grade,
                "Keratin_Pearls_found": match_found
            }

    except Exception as e:
        return {'error': str(e)}
    

def detect_mitotic_figures(image_bytes, template_folder_path):
    try:
        # Read the original image
        original_image = Image.open(BytesIO(image_bytes))

        # Convert image to numpy array
        img = np.array(original_image)
        threshold = 0.62
        best_matching_score = 0
        best_top_left = (0, 0)
        best_bottom_right = (0, 0)
        best_grade = "Normal"

        match_found = False
        for template_file in os.listdir(template_folder_path):
            template_path = os.path.join(template_folder_path, template_file)

            template_image = read_image_from_path(template_path)

            if template_image is not None:
                if len(template_image.shape) == 3:
                    template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

                matched_img, matching_score, top_left, bottom_right = find_matching_template(img, template_image, threshold)

                if matching_score > best_matching_score:
                    best_matching_score = matching_score
                    best_top_left = top_left
                    best_bottom_right = bottom_right
                    main_image_height = img.shape[0]
                    best_grade = get_contour_position(main_image_height, top_left)

                if matched_img is not None:
                    contour_img = img.copy()
                    cv2.rectangle(contour_img, top_left, bottom_right, (0, 255, 0), 2)

                    match_found = True

        if not match_found:
            return {
                "original_image": None,
                "result_image": None,
                "mitotic_figure_score": 0,
                "mitotic_figure_grade": "Normal",
                "mitotic_figure_found": "No presence of mitotic figure observed"
            }
        else:
            contour_img = img.copy()
            cv2.rectangle(contour_img, best_top_left, best_bottom_right, (0, 255, 0), 2)

            # Convert result image to base64 string
            result_image_base64 = image_to_base64(contour_img)
            og_image_base64 = image_to_base64(img)

            # Save the result image to a BytesIO buffer
            result_image_buffer = BytesIO()
            Image.fromarray(contour_img).save(result_image_buffer, format='JPEG')
            result_image_bytes = result_image_buffer.getvalue()
            result_image_base64 = image_to_base64(contour_img)

            return {
                "original_image": og_image_base64,
                "result_image": result_image_base64 if best_matching_score > 0 else None,
                "mitotic_figure_score": best_matching_score,
                "mitotic_figure_grade": best_grade,
                "mitotic_figure_found": match_found
            }

    except Exception as e:
        return {'error': str(e)}
    