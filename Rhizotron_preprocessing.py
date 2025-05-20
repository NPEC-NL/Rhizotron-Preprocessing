"""
Rhizotron Preprocessing

This script processes images from a specified directory to create a high contrast of the Rhizotron using the Red and blue images.
The script also perfroms some simple transoformations to the images to correct for defects in the Scanner.

To use this code you need to first download your images off of the Rhiztron computer and save them locally in their original folder structure.
The script will then process the images and save the results in the same directory as the original images.

The only option the user needs to define is the path to the data directory.
##ADD HOW TO SETUP VIRTUAL ENVIRONMENT

"""
import os
import json
import numpy as np
from glob import glob
from PIL import Image  
from PIL import ImageChops 
from PIL import ImageMath
###################################################################################
#User VARIABLES
data_path = "Topfolder goes here for example 'C:/Rhizotron_Image//G_VDSMO_0300_0800_2IR_1.0'"
############################################################################################

# Get all subfolders in the directory and sort them by timestamp
subfolders = sorted([os.path.join(data_path, folder) for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))])

# ERROR HANDLING:Ensure the number of subfolders is even (pairs of folders)
if len(subfolders) % 2 != 0:
    raise ValueError("The number of subfolders is not even. Ensure the Red and Blue images are being scanned in the settings.")

# Process each pair of subfolders
for i in range(0, len(subfolders), 2):
    folder1 = subfolders[i]
    folder2 = subfolders[i + 1]
    print(f"Processing folder pair: {folder1} and {folder2}")
    # Step 2: Read the metadata.json file from each folder
    #ERROR HANDLING:Check that the two images have the same Treatment ID   
    metadata_file1 = os.path.join(folder1, "metadata.json")
    metadata_file2 = os.path.join(folder2, "metadata.json")

    if not os.path.exists(metadata_file1):
        raise FileNotFoundError(f"metadata.json not found in {folder1}")
    if not os.path.exists(metadata_file2):
        raise FileNotFoundError(f"metadata.json not found in {folder2}")

    with open(metadata_file1, "r") as file:
        meta = json.load(file)
    with open(metadata_file2, "r") as file:
        meta2 = json.load(file)

    if meta["TreatmentId"] != meta2["TreatmentId"]:
        raise ValueError(f"TreatmentId mismatch: {meta['TreatmentId']} (in {folder1}) != {meta2['TreatmentId']} (in {folder2})")

    for folder in [folder1, folder2]:

        #ERROR HANDLING: Check if the metadata exists
        metadata_file = os.path.join(folder, "metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"metadata.json not found in {folder}")

        with open(metadata_file, "r") as file:
            meta = json.load(file)

        # Step 3: Set red_blue based on the ConfigFile string
        config_file = meta["Extra"]["ConfigFile"].lower()
        if "red" in config_file:
            red_blue = True
        elif "blue" in config_file:
            red_blue = False
        else:
            raise ValueError("No Configuration file found for this trial (must contain 'Red' or 'Blue').")

        # Load a single image from the "data" subfolder
        image_folder = os.path.join(folder, "data")
        image_files = glob(os.path.join(image_folder, "*.bmp")) 

        #ERROR HANDLING: Make sure there is only one image in the folder
        if len(image_files) != 1:
            raise ValueError(f"Expected exactly 1 image in {image_folder}, but found {len(image_files)}")
        image_file = image_files[0]  # Get the single image file, I'm sure there is a better way to do this.

        #and assign it red or blue based on the flag
        if red_blue:
            red_image = Image.open(image_files[0])
        else:
            blue_image = Image.open(image_files[0])

    ########################################################################################################
    #  Now we have the red and blue images loaded, we can process them starting with dealing with the red/Blue offset

    #  Translate the blue image in the negative y-axis by 9 pixels
    translation_y = 19
    blue_image = blue_image.transform(
    blue_image.size,  # Keep the original size
    Image.AFFINE,  # Use an affine transformation
    (1, 0, 0, 0, 1, -translation_y)  # Affine matrix for translation (x, y)
    )
    
    #Now create the High Contrast image by dividing the blue image by the red image
    # Convert images to NumPy arrays as doubles for mathematical operations
    red_array = np.asarray(red_image, dtype=np.float64)
    blue_array = np.asarray(blue_image, dtype=np.float64)

    #numerator = red_array - blue_array
   #denominator = red_array + blue_array + 1e-6  
    numerator = blue_array
    denominator = red_array + 1e-6  
    # To avoid glare from those weird hook things, we only normalize on the rhizotron area
    cropped_array = numerator[3500:, 1500:-1500]
    lower_bound = np.percentile(cropped_array, 2)
    upper_bound = np.percentile(cropped_array, 98)
    numerator = np.clip(numerator, lower_bound, upper_bound)

    cropped_array = denominator[3500:, 1500:-1500]
    lower_bound = np.percentile(cropped_array, 2)
    upper_bound = np.percentile(cropped_array, 98)
    denominator = np.clip(denominator, lower_bound, upper_bound)


    result_array = numerator * denominator

    #result_array = np.clip(result_array, lower_bound, upper_bound)

    #Next we need to remove the weird stripes that are caused by the scanner
    cropped_array = result_array[3500:, 1500:-1500]
    row_means = np.mean(cropped_array, axis=0, keepdims=True)
    row_means[row_means == 0] = 1e-6
    result_array[:, 1500:-1500] = result_array[:, 1500:-1500] / row_means

    # Normalize the result to the range [0, 1] using only the rhizotron region.
    cropped_array = result_array[3500:, 1500:-1500]
    lower_bound = np.percentile(cropped_array, 0.5)
    upper_bound = np.percentile(cropped_array, 99.5)
    result_array = (result_array - lower_bound) / (upper_bound - lower_bound)
    result_array = np.clip(result_array, 0, 1)
    result_array = (result_array * 255).astype(np.uint8)

    # Extract PlantId and Datetime from meta
    plant_id = meta["PlantId"]
    datetime_str = meta["Datetime"].replace(":", "_").replace(",", "").replace(" ", "_").replace(".", "_")

    # Build the output filename
    output_name = f"{plant_id}_{datetime_str}.png"
    # Convert the NumPy array back to a Pillow image and save it
    result_image = Image.fromarray(result_array, mode="L")
    output_path = os.path.join(data_path, f"{output_name}_result_image.png")   
    result_image.save(output_path, format="png")

    print(f"Result image saved")
