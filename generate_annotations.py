import os
import glob
from lxml import etree
from PIL import Image

# Directory paths
base_dir = 'C:/Users/darin/Documents/8B/tensorflow/data'
output_dir = 'C:/Users/darin/Documents/8B/tensorflow/annotations'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get list of all subdirectories (each representing a class/pose)
class_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Function to create XML annotation
def create_xml_annotation(img_path, label, output_dir):
    # Load image to get dimensions
    img = Image.open(img_path)
    width, height = img.size

    # Get image file name
    img_file = os.path.basename(img_path)
    img_name, img_ext = os.path.splitext(img_file)

    # Create XML structure
    annotation = etree.Element('annotation')

    folder = etree.SubElement(annotation, 'folder')
    folder.text = label

    filename = etree.SubElement(annotation, 'filename')
    filename.text = img_file

    path = etree.SubElement(annotation, 'path')
    path.text = img_path

    source = etree.SubElement(annotation, 'source')
    database = etree.SubElement(source, 'database')
    database.text = 'Unknown'

    size = etree.SubElement(annotation, 'size')
    width_tag = etree.SubElement(size, 'width')
    width_tag.text = str(width)
    height_tag = etree.SubElement(size, 'height')
    height_tag.text = str(height)
    depth = etree.SubElement(size, 'depth')
    depth.text = '3'

    segmented = etree.SubElement(annotation, 'segmented')
    segmented.text = '0'

    object_tag = etree.SubElement(annotation, 'object')
    name = etree.SubElement(object_tag, 'name')
    name.text = label
    pose = etree.SubElement(object_tag, 'pose')
    pose.text = 'Unspecified'
    truncated = etree.SubElement(object_tag, 'truncated')
    truncated.text = '0'
    difficult = etree.SubElement(object_tag, 'difficult')
    difficult.text = '0'

    bndbox = etree.SubElement(object_tag, 'bndbox')
    xmin = etree.SubElement(bndbox, 'xmin')
    xmin.text = '0'
    ymin = etree.SubElement(bndbox, 'ymin')
    ymin.text = '0'
    xmax = etree.SubElement(bndbox, 'xmax')
    xmax.text = str(width)
    ymax = etree.SubElement(bndbox, 'ymax')
    ymax.text = str(height)

    # Save XML to file
    xml_str = etree.tostring(annotation, pretty_print=True)
    with open(os.path.join(output_dir, img_name + '.xml'), 'wb') as temp_xml:
        temp_xml.write(xml_str)

# Supported image file extensions
img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

# Iterate through each class directory
for class_dir in class_dirs:
    label = class_dir
    img_dir = os.path.join(base_dir, class_dir)
    img_paths = glob.glob(os.path.join(img_dir, '*'))

    # Generate XML annotations for each image
    for img_path in img_paths:
        if os.path.splitext(img_path)[1].lower() in img_extensions:
            create_xml_annotation(img_path, label, output_dir)

print("Annotations generated successfully.")
