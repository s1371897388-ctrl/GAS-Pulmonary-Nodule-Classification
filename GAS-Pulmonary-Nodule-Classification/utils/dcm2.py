import numpy as np
import json
from PIL import Image, ImageDraw
import os
import pydicom
import cv2

def denoise(image_array):
    # 使用高斯滤波去噪
    return cv2.GaussianBlur(image_array, (5, 5), 0)

def normalization(image_array):
    max_value = image_array.max()
    min_value = image_array.min()
    # Normalize
    image_array = (image_array - min_value) / (max_value - min_value) * 255
    # Clip values to [0, 255] range
    image_array = np.clip(image_array, 0, 255)
    # Round to the nearest integer
    image_array = np.round(image_array)
    return image_array.astype(np.uint8)

def crop_nodule(image_array, polygon, padding=10):
    # Find bounding box of the polygon
    min_x = min(polygon, key=lambda x: x[0])[0]
    max_x = max(polygon, key=lambda x: x[0])[0]
    min_y = min(polygon, key=lambda x: x[1])[1]
    max_y = max(polygon, key=lambda x: x[1])[1]

    # Add padding to the bounding box
    min_x = max(min_x - padding, 0)
    max_x = min(max_x + padding, image_array.shape[1])
    min_y = max(min_y - padding, 0)
    max_y = min(max_y + padding, image_array.shape[0])

    # Crop the image
    cropped_image = image_array[min_y:max_y, min_x:max_x]
    # Resize to 64x64 with Lanczos interpolation
    cropped_image = np.array(Image.fromarray(cropped_image).resize((64, 64), resample=Image.LANCZOS))

    return cropped_image

def process_nodule(dicom_path, json_path, output_dir):
    print(f"Processing DICOM: {dicom_path}")
    print(f"Using JSON: {json_path}")

    # Read DICOM file
    dicom = pydicom.dcmread(dicom_path)
    image_array = dicom.pixel_array.astype(np.float32)

    # Denoise
    denoised_image_array = denoise(image_array)

    # Normalize
    normalized_image_array = normalization(denoised_image_array)

    # Convert to RGB
    normalized_image_array_rgb = np.stack((normalized_image_array,) * 3, axis=-1)

    # Load JSON file
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # Process each shape in JSON
    for i, shape in enumerate(json_data['shapes']):
        points = shape['points']
        polygon = [(int(point[0]), int(point[1])) for point in points]

        # Crop nodule
        cropped_image = crop_nodule(normalized_image_array_rgb, polygon)

        # Save cropped image as JPG
        output_path = os.path.join(output_dir, f'{os.path.basename(dicom_path).replace(".dcm", f"_cropped_{i}.jpg")}')
        Image.fromarray(cropped_image).save(output_path)
        print(f"Saved cropped image to: {output_path}")

# Main script to process all subfolders in E:\privatedata
base_dir = 'E:/privatedata'

for folder in range(1, 20):
    folder_path = os.path.join(base_dir, str(folder))
    dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
    json_folder_path = os.path.join(folder_path, 'json')
    json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(json_folder_path, json_file)
        for dicom_file in dicom_files:
            dicom_path = os.path.join(folder_path, dicom_file)
            process_nodule(dicom_path, json_path, folder_path)

# 日期：2024/6/16
