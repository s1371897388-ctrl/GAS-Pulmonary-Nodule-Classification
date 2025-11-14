import json
import cv2
import os
from PIL import Image
import numpy as np

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def crop_nodule_from_image(image_path, json_data):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Extract polygon coordinates from JSON
    points = json_data['shapes'][0]['points']
    polygon = [(int(point[0]), int(point[1])) for point in points]

    # Find bounding box of the polygon
    min_x = min(polygon, key=lambda x: x[0])[0]
    max_x = max(polygon, key=lambda x: x[0])[0]
    min_y = min(polygon, key=lambda x: x[1])[1]
    max_y = max(polygon, key=lambda x: x[1])[1]

    # Calculate the center and the side length of the square
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    side_length = max(max_x - min_x, max_y - min_y)

    # Calculate the coordinates of the square's top-left and bottom-right corners
    square_xmin = max(center_x - side_length // 2, 0)
    square_ymin = max(center_y - side_length // 2, 0)
    square_xmax = min(center_x + side_length // 2, image.shape[1])
    square_ymax = min(center_y + side_length // 2, image.shape[0])

    # Crop the square region
    square_image = image[square_ymin:square_ymax, square_xmin:square_xmax]

    # Resize to 64x64
    resized_image = cv2.resize(square_image, (64, 64), interpolation=cv2.INTER_LANCZOS4)

    return resized_image


def save_image(image, output_path):
    Image.fromarray(image).save(output_path)


def process_nodule(image_path, json_path, output_dir):
    json_data = load_json(json_path)
    cropped_image = crop_nodule_from_image(image_path, json_data)

    output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_cropped.jpg'))
    save_image(cropped_image, output_path)

    return output_path


# Example usage
image_path = 'E:/test/1/317.jpg'
json_path = 'E:/test/1/317.json'
output_dir = 'E:/test/1/image'

output_path = process_nodule(image_path, json_path, output_dir)
print(f"Cropped image saved to: {output_path}")

# dicom_path = 'E:/test/1/dicom/1.3.12.2.1107.5.1.4.66552.30000017112100233688500001636.dcm'
# json_path = 'E:/test/1/317.json'


# 作者：孙海滨
# 日期：2024/6/16
