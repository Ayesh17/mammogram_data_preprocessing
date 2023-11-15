import os
import pydicom
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

import config


def convert_xml_roi_to_png(dicom_folder, xml_folder, png_folder):
    os.makedirs(png_folder, exist_ok=True)

    for filename in os.listdir(dicom_folder):
        if filename.endswith(".dcm"):
            file_path = os.path.join(dicom_folder, filename)
            ds = pydicom.dcmread(file_path)


            # Load the corresponding .xml file
            xml_filename = ((os.path.splitext(filename)[0]).split('_'))[0] + ".xml"
            xml_path = os.path.join(xml_folder, xml_filename)
            print("xml_path", xml_path)

            if os.path.exists(xml_path):
                # Read the XML file and extract ROI coordinates
                roi_coordinates = read_xml_roi(xml_path)
                print("roi_coordinates", roi_coordinates)

                if roi_coordinates:
                    # Draw ROI on the DICOM image
                    image_with_roi = draw_roi(ds.pixel_array, roi_coordinates)

                    # Convert to PIL Image for saving as PNG
                    roi_image = Image.fromarray(image_with_roi)

                    # Save the image with ROI as PNG
                    output_filename = os.path.splitext(filename)[0] + "_roi.png"
                    print("output_filename", output_filename)
                    output_path = os.path.join(png_folder, output_filename)
                    roi_image.save(output_path)

def read_xml_roi(xml_file_path):
    # Parse the XML file and extract ROI coordinates or information
    roi_coordinates = []

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Assuming the ROI coordinates are stored under 'ROI' tags
    # Modify this part according to your XML structure
    for roi in root.findall('.//ROI'):
        # Extract relevant information such as coordinates, shapes, etc.
        # Modify this part according to your XML structure
        x = float(roi.find('x_coordinate').text)
        y = float(roi.find('y_coordinate').text)
        width = float(roi.find('width').text)
        height = float(roi.find('height').text)

        # Store the extracted information in a suitable format
        roi_info = {'x': x, 'y': y, 'width': width, 'height': height}
        roi_coordinates.append(roi_info)

    return roi_coordinates

def draw_roi(image_array, roi_coordinates):
    # Use the ROI coordinates to draw the ROI on the image array
    # Implement logic to draw ROI on the DICOM image array
    # Return the image array with ROI drawn
    image_with_roi = image_array.copy()

    # Use ImageDraw to draw the ROI on the image array
    draw = ImageDraw.Draw(image_with_roi)

    # Implement drawing based on the extracted ROI coordinates

    return image_with_roi

# Replace with your DICOM folder, XML folder, and output PNG folder paths
dicom_folder_path = os.path.join(config.INBREAST_PATH, "AllDICOMs")
xml_folder_path = os.path.join(config.INBREAST_PATH, "AllXML")
png_output_folder_path = os.path.join(config.INBREAST_PATH, "Converted_ROI")


convert_xml_roi_to_png(dicom_folder_path, xml_folder_path, png_output_folder_path)
