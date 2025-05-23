import os
import csv
import glob
import shutil
from datetime import datetime

from utils.inference_localization import localization_and_crop_image
from utils.number_recognition import get_number_from_image


image_path = 'test_image.png'
localization_and_crop_image(image_path)

cropped_objects_dir = 'cropped_objects'
subdirs = ['SYS', 'DLA', 'PUL']

# Dictionary to store numbers from each subdirectory
results = {}

# Process each subdirectory (SYS, DLA, PUL)
for subdir in subdirs:
    subdir_path = os.path.join(cropped_objects_dir, subdir)
    
    # Check if subdirectory exists
    if not os.path.exists(subdir_path):
        print(f"Directory {subdir_path} does not exist, skipping.")
        continue
    
    # Get all image files in the subdirectory
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(subdir_path, ext)))
    
    # Check if any images were found
    if not image_files:
        print(f"No images found in {subdir_path}, skipping.")
        continue
    
    # Sort images to process them in order
    image_files.sort()
    
    # Process the first image to get the number
    number = get_number_from_image(image_files[0])
    results[subdir] = number

# Add current timestamp to results
results['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Save results to CSV with append mode
csv_file = 'bp_readings.csv'
file_exists = os.path.isfile(csv_file)

# Define fieldnames with Timestamp first
fieldnames = ['Timestamp'] + subdirs

with open(csv_file, 'a', newline='') as f:  # 'a' for append mode
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:  # Only write header if file doesn't exist
        writer.writeheader()
    writer.writerow(results)

print(f"Results appended to {csv_file}")

# Delete the cropped_objects directory and all its contents
if os.path.exists(cropped_objects_dir):
    shutil.rmtree(cropped_objects_dir)
    print(f"Deleted {cropped_objects_dir} directory")