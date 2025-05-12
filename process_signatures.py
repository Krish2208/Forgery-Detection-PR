import os
import json
import cv2
import numpy as np
from feature_extraction import extract_signature_features

def process_signature_folder(input_folder, output_folder, cropped_folder):
    """
    Process all signature images in a folder, extract features, and save results.
    
    Parameters:
    -----------
    input_folder : str
        Path to the folder containing signature images
    output_folder : str
        Path to save the extracted features and cropped images
    """
    # Create output directories if they don't exist
    feature_output_dir = output_folder
    cropped_output_dir = cropped_folder
    
    os.makedirs(feature_output_dir, exist_ok=True)
    os.makedirs(cropped_output_dir, exist_ok=True)

    
    # Get all image files in the input folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder) 
                   if os.path.isfile(os.path.join(input_folder, f)) 
                   and f.lower().endswith(image_extensions)]
    
    print(f"Found {len(image_files)} image files to process")
    
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        
        # Get filename without extension for output naming
        base_filename = os.path.splitext(image_file)[0]
        
        # Process the image
        try:
            # Step 1: Load the image
            original_img = cv2.imread(input_path)
            if original_img is None:
                print(f"Error: Could not read image {input_path}")
                continue
            
            # Step 2: Convert to grayscale for finding the bounding box
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            
            # Step 3: Binarize the image to isolate the signature
            _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
            
            # Step 4: Find bounding box by locating extreme black pixels
            # Find rows and columns that contain signature pixels (non-zero pixels in binary image)
            rows_with_signature = np.where(np.any(binary > 0, axis=1))[0]
            cols_with_signature = np.where(np.any(binary > 0, axis=0))[0]
            
            if len(rows_with_signature) == 0 or len(cols_with_signature) == 0:
                print(f"Warning: No signature pixels found in {image_file}")
                # Use the original image
                cropped_resized = cv2.resize(original_img, (512, 256))
            else:
                # Get the extremes to form bounding box
                y_min = rows_with_signature[0]
                y_max = rows_with_signature[-1]
                x_min = cols_with_signature[0]
                x_max = cols_with_signature[-1]
                
                # Calculate width and height
                w = x_max - x_min + 1
                h = y_max - y_min + 1
                
                # Add a small padding (5% of dimensions) to the bounding box
                padding_x = int(w * 0.05)
                padding_y = int(h * 0.05)
                
                # Ensure padded coordinates stay within image boundaries
                x_start = max(0, x_min - padding_x)
                y_start = max(0, y_min - padding_y)
                x_end = min(gray.shape[1], x_max + padding_x + 1)
                y_end = min(gray.shape[0], y_max + padding_y + 1)
                
                # Crop the image to the padded bounding box
                cropped = original_img[y_start:y_end, x_start:x_end]
                
                # Resize the cropped image to 512x256
                cropped_resized = cv2.resize(cropped, (512, 256))
            
            # Save the cropped and resized image
            cropped_image_path = os.path.join(cropped_output_dir, f"{base_filename}_cropped.png")
            cv2.imwrite(cropped_image_path, cropped_resized)
            
            # Extract features using the imported function
            features = extract_signature_features(cropped_image_path)
            
            # Save the features to a JSON file
            feature_output_path = os.path.join(feature_output_dir, f"{base_filename}_features.json")
            with open(feature_output_path, 'w') as f:
                json.dump(features, f, indent=2)
            
            print(f"Processed {image_file} - features saved to {feature_output_path}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
    
    print(f"Processing complete. Processed {len(image_files)} images.")
    print(f"Cropped images saved to: {cropped_output_dir}")
    print(f"Features saved to: {feature_output_dir}")

if __name__ == "__main__":
    
    process_signature_folder("signatures/full_forg", "signatures/features_forg", "signatures/cropped_forg")
    process_signature_folder("signatures/full_org", "signatures/features_org", "signatures/cropped_org")
