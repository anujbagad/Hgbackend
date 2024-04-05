# cellsize.py 
import os
import cv2
import numpy as np
from flask import jsonify, request
import base64
import json

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

# def calculate_average_of_dataset(dataset_path):
#     averages = {'Top': [], 'Middle': [], 'Bottom': []}

#     for subdir in os.listdir(dataset_path):
#         subdir_path = os.path.join(dataset_path, subdir)

#         if os.path.isdir(subdir_path):
#             subdir_averages = {'Top': [], 'Middle': [], 'Bottom': []}

#             for file_name in os.listdir(subdir_path):
#                 file_path = os.path.join(subdir_path, file_name)

#                 # Read and process each image in the dataset
#                 image = cv2.imread(file_path, cv2.IMREAD_COLOR)
#                 gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 _, nuclei_count, nuclei_sizes, nuclei_contours = find_draw_nuclei_boundaries_and_get_sizes(
#                     gray_image, min_area=20
#                 )
#                 image_height, _, _ = image.shape
#                 avg_top, avg_mid, avg_bottom = calculate_average_nucleus_size(
#                     image_height, nuclei_contours, nuclei_sizes
#                 )

#                 subdir_averages['Top'].append(avg_top)
#                 subdir_averages['Middle'].append(avg_mid)
#                 subdir_averages['Bottom'].append(avg_bottom)

#             avg_top_subdir = np.mean(subdir_averages['Top']) if any(subdir_averages['Top']) else 0
#             avg_mid_subdir = np.mean(subdir_averages['Middle']) if any(subdir_averages['Middle']) else 0
#             avg_bottom_subdir = np.mean(subdir_averages['Bottom']) if any(subdir_averages['Bottom']) else 0

#             print(f'{subdir} Averages:')
#             print({'Top': avg_top_subdir, 'Mid': avg_mid_subdir, 'Bottom': avg_bottom_subdir})

#             averages['Top'].append(avg_top_subdir)
#             averages['Middle'].append(avg_mid_subdir)
#             averages['Bottom'].append(avg_bottom_subdir)

#     return avg_top_subdir, avg_mid_subdir, avg_bottom_subdir



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
