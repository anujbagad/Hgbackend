import cv2
import numpy as np
import os
from PIL import Image
from io import BytesIO
import base64

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

# Function to convert image to base64 string
def image_to_base64(image):
    pil_image = Image.fromarray(image)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"
