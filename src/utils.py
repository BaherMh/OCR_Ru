import base64
import os

import pandas as pd


def exctract_images(tsv_data, save_dir):

    df = pd.read_csv(tsv_data, sep='\t')

    # Create a directory to save images (optional)
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over each row
    for _, row in df.iterrows():
        index = row['index']
        image_base64 = row['image']
        
        # Skip if image data is missing
        if pd.isna(image_base64) or image_base64.strip() == '':
            continue
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            print(f"Failed to decode image for index {index}: {e}")
            continue
        
        # Save image
        image_path = f"{save_dir}/{index}.png"  # assuming PNG; adjust extension if needed
        with open(image_path, 'wb') as f:
            f.write(image_data)

def unify_string_format(text):
    return text.strip().lower().replace('\n',' ').replace(' ', '')
