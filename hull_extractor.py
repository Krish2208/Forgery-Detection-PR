import numpy as np
import cv2
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import argparse
import os

def extract_signature_boundary(image_path, output_path=None, show_plot=False, threshold=220):
    """
    Extract the convex hull (boundary) of a signature from an image.
    
    Parameters:
    -----------
    image_path : str
        Path to the signature image
    output_path : str, optional
        Path to save the output visualization
    show_plot : bool, default=False
        Whether to display the plot
    threshold : int, default=220
        Threshold value for binarization (0-255)
        
    Returns:
    --------
    hull_points : numpy.ndarray
        Points of the convex hull boundary
    hull_area : float
        Area of the convex hull
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to binarize the image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Find coordinates of non-zero pixels (signature points)
    points = np.column_stack(np.where(binary > 0))
    
    # If no signature points found, return empty results
    if len(points) < 3:  # Need at least 3 points for a convex hull
        print("Warning: No signature points detected. Try adjusting the threshold.")
        return np.array([]), 0.0
    
    # Compute the convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_area = hull.volume  # In 2D, volume is actually the area
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Plot original image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot binary image
    plt.subplot(2, 2, 2)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')
    
    # Plot signature points and hull
    plt.subplot(2, 2, 3)
    plt.scatter(points[:, 1], points[:, 0], s=0.5, c='blue', alpha=0.5)
    plt.plot(hull_points[:, 1], hull_points[:, 0], 'r-')
    plt.plot(hull_points[[0, -1], 1], hull_points[[0, -1], 0], 'r-')
    plt.title('Signature Points and Convex Hull')
    plt.gca().invert_yaxis()
    plt.axis('off')
    
    # Plot hull only with metrics
    plt.subplot(2, 2, 4)
    plt.plot(hull_points[:, 1], hull_points[:, 0], 'r-')
    plt.plot(hull_points[[0, -1], 1], hull_points[[0, -1], 0], 'r-')
    plt.scatter(hull_points[:, 1], hull_points[:, 0], s=10, c='red')
    plt.title(f'Convex Hull (Area: {hull_area:.2f} pixels)')
    plt.gca().invert_yaxis()
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save or show the visualization
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return hull_points, hull_area

def extract_features(hull_points, hull_area, img_shape):
    """
    Extract features from the signature convex hull for pattern recognition.
    
    Parameters:
    -----------
    hull_points : numpy.ndarray
        Points of the convex hull boundary
    hull_area : float
        Area of the convex hull
    img_shape : tuple
        Shape of the original image (height, width)
        
    Returns:
    --------
    features : dict
        Dictionary of extracted features
    """
    if len(hull_points) < 3:
        return {
            'hull_area': 0,
            'hull_perimeter': 0,
            'aspect_ratio': 1,
            'occupancy_ratio': 0,
            'centroid_x': 0,
            'centroid_y': 0,
            'num_vertices': 0
        }
    
    # Calculate perimeter of the convex hull
    perimeter = 0
    for i in range(len(hull_points)):
        j = (i + 1) % len(hull_points)
        perimeter += np.sqrt(np.sum((hull_points[i] - hull_points[j])**2))
    
    # Calculate bounding box
    min_y, min_x = np.min(hull_points, axis=0)
    max_y, max_x = np.max(hull_points, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Calculate aspect ratio
    aspect_ratio = width / height if height > 0 else 1
    
    # Calculate occupancy ratio (hull area / image area)
    image_area = img_shape[0] * img_shape[1]
    occupancy_ratio = hull_area / image_area
    
    # Calculate centroid
    centroid_y = np.mean(hull_points[:, 0])
    centroid_x = np.mean(hull_points[:, 1])
    
    # Normalized centroid position (0-1 range)
    norm_centroid_x = centroid_x / img_shape[1]
    norm_centroid_y = centroid_y / img_shape[0]
    
    return {
        'hull_area': hull_area,
        'hull_perimeter': perimeter,
        'aspect_ratio': aspect_ratio,
        'occupancy_ratio': occupancy_ratio,
        'centroid_x': norm_centroid_x,
        'centroid_y': norm_centroid_y,
        'num_vertices': len(hull_points)
    }

def main():
    parser = argparse.ArgumentParser(description='Extract the convex hull of a signature from an image')
    parser.add_argument('image_path', type=str, help='Path to the signature image')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to save visualization')
    parser.add_argument('--show', '-s', action='store_true', help='Show the visualization')
    parser.add_argument('--threshold', '-t', type=int, default=220, help='Threshold for binarization (0-255)')
    parser.add_argument('--features', '-f', action='store_true', help='Print extracted features')
    
    args = parser.parse_args()
    
    try:
        # Get image shape for feature extraction
        img = cv2.imread(args.image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {args.image_path}")
        img_shape = img.shape
        
        # Extract the convex hull
        hull_points, hull_area = extract_signature_boundary(
            args.image_path, 
            args.output, 
            args.show,
            args.threshold
        )
        
        if args.features and len(hull_points) >= 3:
            features = extract_features(hull_points, hull_area, img_shape)
            print("\nExtracted Features:")
            for key, value in features.items():
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
