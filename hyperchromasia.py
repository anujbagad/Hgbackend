import numpy as np
import skimage.io
import requests
from PIL import Image
from io import BytesIO
import histomicstk as htk

import base64

def detect_hyperchromasia(image_bytes):
    try:
        # Read the original image
        original_image = Image.open(BytesIO(image_bytes))

        # Convert image to numpy array
        img = np.array(original_image)

        # Load reference image for normalization
        ref_image_file = './mod_ref.jpg'  # Update the reference image file path
        im_reference = skimage.io.imread(ref_image_file)[:, :, :3]

        # get mean and stddev of the reference image in lab space
        mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)

        # perform Reinhard color normalization
        im_nmzd = htk.preprocessing.color_normalization.reinhard(img, mean_ref, std_ref)

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
        im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_nmzd, W).Stains

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
        elif avg_intensity_bottom_section > avg_intensity_middle_section and avg_intensity_bottom_section > avg_intensity_top_section:
            classification = "Mild"
        else:
            classification = "Unknown"

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


def image_to_base64(image):
    # Convert image to PIL format
    pil_image = Image.fromarray(image)
    # Convert PIL image to base64 string
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


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
