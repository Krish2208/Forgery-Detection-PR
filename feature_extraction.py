import cv2
import numpy as np
from scipy.spatial import ConvexHull
from skimage.feature import local_binary_pattern, hog

LBP_POINTS = 24
LBP_RADIUS = 8
LBP_NUM_BINS = LBP_POINTS + 2
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
HOG_INPUT_SIZE = (256, 128)
EXPECTED_HOG_FEATURES_LEN = 3780

def extract_lbp_features(image_gray):
    """Extract Local Binary Pattern features."""
    try:
        # Ensure image is 2D
        if image_gray.ndim == 3:
            image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

        lbp = local_binary_pattern(image_gray, LBP_POINTS, LBP_RADIUS, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_NUM_BINS + 1), range=(0, LBP_NUM_BINS))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # Normalize
        return hist.tolist()
    except Exception as e:
        print(f"Warning: LBP feature extraction failed: {e}")
        return [0.0] * LBP_NUM_BINS
    
def extract_hog_features(image_gray):
    """Extract Histogram of Oriented Gradients features."""
    try:
        # Ensure image is 2D
        if image_gray.ndim == 3:
            image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

        # Resize for consistent HOG feature vector length
        img_resized_for_hog = cv2.resize(image_gray, HOG_INPUT_SIZE)

        hog_feats = hog(img_resized_for_hog, orientations=HOG_ORIENTATIONS,
                        pixels_per_cell=HOG_PIXELS_PER_CELL,
                        cells_per_block=HOG_CELLS_PER_BLOCK,
                        block_norm='L2-Hys', visualize=False, feature_vector=True)
        # Pad or truncate if necessary, though resizing should make it fixed.
        if len(hog_feats) < EXPECTED_HOG_FEATURES_LEN:
            hog_feats = np.pad(hog_feats, (0, EXPECTED_HOG_FEATURES_LEN - len(hog_feats)), 'constant')
        elif len(hog_feats) > EXPECTED_HOG_FEATURES_LEN:
            hog_feats = hog_feats[:EXPECTED_HOG_FEATURES_LEN]
        return hog_feats.tolist()
    except Exception as e:
        print(f"Warning: HOG feature extraction failed: {e}")
        return [0.0] * EXPECTED_HOG_FEATURES_LEN

def extract_signature_features(image_path, output_path=None, show_plot=False, 
                              threshold=220, use_sift=True, use_transitions=True,
                              Tr=20, Th=20, Tw=20, num_transition_features=40,
                              use_lbp=True, use_hog=True):
    """
    Comprehensive function to extract multiple feature sets from a signature image.
    
    Parameters:
    -----------
    image_path : str
        Path to the signature image
    output_path : str, optional
        Path to save the visualization output
    show_plot : bool, default=False
        Whether to display the visualization plot
    threshold : int, default=220
        Threshold value for binarization (0-255)
    use_sift : bool, default=True
        Whether to include SIFT features
    use_transitions : bool, default=True
        Whether to include transition features
    Tr, Th, Tw : int
        Parameters for geometric feature extraction
    num_transition_features : int
        Number of transition features to extract
    use_lbp : bool, default=True
        Whether to include LBP features
    use_hog : bool, default=True
        Whether to include HOG features
        
    Returns:
    --------
    dict
        Dictionary containing all extracted features
    """
    # 1. Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    height, width = gray.shape
    
    # Initialize the features dictionary
    features = {}
    
    # 2. Extract boundary and geometric features
    boundary_features = extract_signature_boundary(img, binary, show_plot, output_path)
    features.update(boundary_features)
    
    # 3. Extract geometric features
    geo_features = extract_geometric_features(binary, Tr, Th, Tw)
    features.update(geo_features)
    
    # 4. Extract sixfold surface features
    sixfold_features = extract_sixfold_surface_feature(binary)
    features['sixfold_surface'] = sixfold_features
    
    # 5. Conditionally extract transition features
    if use_transitions:
        transition_features = extract_transition_features(binary, num_transition_features)
        features['transition_features'] = transition_features.tolist()
    
    # 6. Conditionally extract SIFT features
    if use_sift:
        try:
            keypoints, descriptors = compute_sift(image_path)
            # Store number of keypoints and mean of descriptors if available
            features['num_keypoints'] = len(keypoints)
            if descriptors is not None and len(descriptors) > 0:
                features['mean_sift_descriptor'] = np.mean(descriptors, axis=0).tolist()
            else:
                features['mean_sift_descriptor'] = []
        except Exception as e:
            print(f"Warning: SIFT feature extraction failed: {e}")
            features['num_keypoints'] = 0
            features['mean_sift_descriptor'] = []
            
    # 7. Conditionally extract LBP features
    if use_lbp:
        try:
            lbp_features = extract_lbp_features(gray)
            features['lbp_features'] = lbp_features
        except Exception as e:
            print(f"Warning: LBP feature extraction failed: {e}")
            features['lbp_features'] = [0.0] * LBP_NUM_BINS
    
    # 8. Conditionally extract HOG features
    if use_hog:
        try:
            hog_features = extract_hog_features(gray)
            features['hog_features'] = hog_features
        except Exception as e:
            print(f"Warning: HOG feature extraction failed: {e}")
            features['hog_features'] = [0.0] * EXPECTED_HOG_FEATURES_LEN
    
    return features

def extract_signature_boundary(img, binary, show_plot=False, output_path=None):
    """
    Extract the convex hull (boundary) and related features of a signature from a binary image.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Original image
    binary : numpy.ndarray
        Binary image of the signature
    show_plot : bool
        Whether to show visualization
    output_path : str, optional
        Path to save visualization
        
    Returns:
    --------
    dict
        Dictionary of boundary-related features
    """
    height, width = img.shape[:2]
    
    # Find coordinates of non-zero pixels (signature points)
    points = np.column_stack(np.where(binary > 0))
    
    # If no signature points found, return default values
    if len(points) < 3:  # Need at least 3 points for a convex hull
        print("Warning: No signature points detected. Try adjusting the threshold.")
        return {
            'convexity': 0,
            'signature_density': 0,
            'compactness': 0,
            'eccentricity': 0,
            'hu_moments': [0] * 7
        }
    
    # Compute the convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_area = hull.volume  # In 2D, volume is actually the area
    
    # Extract contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signature_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(signature_contour)
    contour_area = cv2.contourArea(signature_contour)
    
    # Calculate features
    convexity = contour_area / hull_area if hull_area > 0 else 0
    image_area = height * width
    signature_density = contour_area / image_area
    
    # Calculate perimeter of the convex hull
    hull_perimeter = 0
    for i in range(len(hull_points)):
        j = (i + 1) % len(hull_points)
        hull_perimeter += np.sqrt(np.sum((hull_points[i] - hull_points[j])**2))
    
    # Compactness: normalized ratio of perimeter^2 to area (scale-invariant)
    compactness = (hull_perimeter ** 2) / (4 * np.pi * hull_area) if hull_area > 0 else 0
    
    # Aspect ratio and eccentricity from oriented bounding box
    width_rect, height_rect = rect[1]
    
    # Make sure width is the longer dimension
    if width_rect < height_rect:
        width_rect, height_rect = height_rect, width_rect
    
    # Eccentricity calculation
    if height_rect > 0:
        semi_major = width_rect / 2
        semi_minor = height_rect / 2
        eccentricity = np.sqrt(1 - (semi_minor**2 / semi_major**2)) if semi_major > 0 else 0
    else:
        eccentricity = 1  # Degenerate case
    
    # Calculate Hu moments for the contour
    mask = np.zeros((binary.shape[0], binary.shape[1]), dtype=np.uint8)
    contour_reshaped = np.array([np.flip(pt) for pt in hull_points])
    contour_reshaped = contour_reshaped.reshape(-1, 1, 2)
    cv2.drawContours(mask, [contour_reshaped], 0, 255, -1)
    
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return {
        'convexity': convexity,
        'signature_density': signature_density,
        'compactness': compactness,
        'eccentricity': eccentricity,
        'hu_moments': hu_moments.tolist()
    }

def count_black_pixels_on_ray(binary_image, center_x, center_y, angle, prev_angle=None):
    """
    Count the number of black pixels along a ray from the center point.
    
    Parameters:
    -----------
    binary_image : numpy.ndarray
        Binary image where signature pixels are white (255)
    center_x, center_y : int
        Center coordinates of the ray
    angle : float
        Angle of the ray in radians
    prev_angle : float, optional
        Previous angle for transition counting
        
    Returns:
    --------
    int
        Number of black pixels crossed
    """
    height, width = binary_image.shape
    
    # Calculate direction vector
    dx = np.cos(angle)
    dy = np.sin(angle)
    
    # Determine max ray length (to edge of image)
    if dx == 0:
        max_t = (height - center_y) / dy if dy > 0 else center_y / (-dy)
    elif dy == 0:
        max_t = (width - center_x) / dx if dx > 0 else center_x / (-dx)
    else:
        tx1 = (width - center_x) / dx if dx > 0 else center_x / (-dx)
        ty1 = (height - center_y) / dy if dy > 0 else center_y / (-dy)
        max_t = min(tx1, ty1)
    
    # Sample points along the ray
    count = 0
    prev_value = None
    
    # Use Bresenham-like algorithm for faster ray traversal
    num_samples = int(max_t) + 1
    for t in range(num_samples):
        # Scale t to ensure we reach the edge
        t_scaled = t * max_t / num_samples
        
        # Calculate sample point
        x = int(center_x + dx * t_scaled)
        y = int(center_y + dy * t_scaled)
        
        # Ensure we're inside the image
        if 0 <= x < width and 0 <= y < height:
            current_value = binary_image[y, x]
            
            # Count transitions
            if prev_value is not None and prev_value != current_value:
                count += 1
            
            prev_value = current_value
    
    return count

def measure_heights(binary_image, pos_x, center_y):
    """
    Measure top and bottom heights from a given position.
    
    Parameters:
    -----------
    binary_image : numpy.ndarray
        Binary image where signature pixels are white (255)
    pos_x : int
        X-coordinate of the measuring point
    center_y : int
        Y-coordinate of the reference point
        
    Returns:
    --------
    tuple
        (top_height, bottom_height)
    """
    height, width = binary_image.shape
    
    if pos_x < 0 or pos_x >= width:
        return 0, 0
    
    # Measure upward
    top_height = 0
    for y in range(center_y, -1, -1):
        if binary_image[y, pos_x] > 0:
            top_height = center_y - y
    
    # Measure downward
    bottom_height = 0
    for y in range(center_y, height):
        if binary_image[y, pos_x] > 0:
            bottom_height = y - center_y
    
    return top_height, bottom_height

def measure_widths(binary_image, center_x, pos_y):
    """
    Measure left and right widths from a given position.
    
    Parameters:
    -----------
    binary_image : numpy.ndarray
        Binary image where signature pixels are white (255)
    center_x : int
        X-coordinate of the reference point
    pos_y : int
        Y-coordinate of the measuring point
        
    Returns:
    --------
    tuple
        (left_width, right_width)
    """
    height, width = binary_image.shape
    
    if pos_y < 0 or pos_y >= height:
        return 0, 0
    
    # Measure leftward
    left_width = 0
    for x in range(center_x, -1, -1):
        if binary_image[pos_y, x] > 0:
            left_width = center_x - x
    
    # Measure rightward
    right_width = 0
    for x in range(center_x, width):
        if binary_image[pos_y, x] > 0:
            right_width = x - center_x
    
    return left_width, right_width

def extract_geometric_features(binary_image, Tr=20, Th=20, Tw=20):
    """
    Extract geometric features from a signature image using both polar and Cartesian coordinates.
    
    Parameters:
    -----------
    binary_image : numpy.ndarray
        Binary image where signature pixels are white (255)
    Tr, Th, Tw : int
        Parameters for equidistant sampling
        
    Returns:
    --------
    dict
        Dictionary containing geometric features
    """
    height, width = binary_image.shape
    
    # Find contours to extract envelope
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("Warning: No signature contour found in the image")
        # Return default empty features
        return {
            'polar_features': np.zeros((Tr, 3)).tolist(),
            'top_heights': np.zeros(Th).tolist(),
            'bottom_heights': np.zeros(Th).tolist(),
            'left_widths': np.zeros(Tw).tolist(),
            'right_widths': np.zeros(Tw).tolist()
        }
    
    # Get the largest contour (assuming it's the signature)
    signature_contour = max(contours, key=cv2.contourArea)
    
    # Find geometric center
    M = cv2.moments(signature_contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(signature_contour)
        center_x = x + w // 2
        center_y = y + h // 2
    
    # Get bounding box for envelope measurements
    x, y, w, h = cv2.boundingRect(signature_contour)
    
    # Extract polar-coordinate features
    contour_points = signature_contour.reshape(-1, 2)
    
    # Sort contour points by angle for proper envelope representation
    angles = np.arctan2(contour_points[:, 1] - center_y, contour_points[:, 0] - center_x)
    sorted_indices = np.argsort(angles)
    sorted_contour = contour_points[sorted_indices]
    
    # Get equidistant samples
    T = len(sorted_contour)
    p = max(1, int(T / Tr))  # Ensure p is at least 1
    
    polar_features = []
    prev_radius = None
    prev_angle = None
    
    for t in range(Tr):
        idx = (t * p) % T
        x, y = sorted_contour[idx]
        
        # Calculate radius and angle
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        angle = np.arctan2(y - center_y, x - center_x)
        
        # For the first point, use the last point as the previous
        if t == 0:
            prev_idx = ((Tr - 1) * p) % T
            prev_x, prev_y = sorted_contour[prev_idx]
            prev_radius = np.sqrt((prev_x - center_x)**2 + (prev_y - center_y)**2)
            prev_angle = np.arctan2(prev_y - center_y, prev_x - center_x)
        
        # Calculate radius derivative
        radius_derivative = radius - prev_radius if prev_radius is not None else 0
        
        # Count black pixels crossed
        black_pixels = count_black_pixels_on_ray(binary_image, center_x, center_y, angle, prev_angle)
        
        polar_features.append([radius_derivative, angle, black_pixels])
        
        # Update previous values
        prev_radius = radius
        prev_angle = angle
    
    # Extract Cartesian coordinate features
    top_heights = []
    bottom_heights = []
    
    for t in range(Th):
        pos_x = x + int(t * w / (Th - 1)) if Th > 1 else x + w // 2
        top_height, bottom_height = measure_heights(binary_image, pos_x, center_y)
        top_heights.append(top_height)
        bottom_heights.append(bottom_height)
    
    left_widths = []
    right_widths = []
    
    for t in range(Tw):
        pos_y = y + int(t * h / (Tw - 1)) if Tw > 1 else y + h // 2
        left_width, right_width = measure_widths(binary_image, center_x, pos_y)
        left_widths.append(left_width)
        right_widths.append(right_width)
    
    return {
        'polar_features': np.array(polar_features).tolist(),
        'top_heights': np.array(top_heights).tolist(),
        'bottom_heights': np.array(bottom_heights).tolist(),
        'left_widths': np.array(left_widths).tolist(),
        'right_widths': np.array(right_widths).tolist()
    }

def extract_transition_features(binary_image, num_features=40):
    """
    Extract transition features by recording the locations of transitions
    between foreground and background pixels.
    
    Parameters:
    -----------
    binary_image : numpy.ndarray
        Binary image where signature pixels are white (255)
    num_features : int
        Number of features to extract
        
    Returns:
    --------
    numpy.ndarray
        Array of normalized transition positions
    """
    height, width = binary_image.shape
    transitions = []
    
    # Left to right transitions
    for i in range(height):
        row_transitions = np.where(np.diff(binary_image[i, :]) != 0)[0]
        # Normalize transition positions
        norm_positions = (row_transitions + 1) / width  # +1 because diff reduces length by 1
        transitions.extend(norm_positions)
    
    # Right to left transitions
    for i in range(height):
        row_flipped = np.flip(binary_image[i, :])
        row_transitions = np.where(np.diff(row_flipped) != 0)[0]
        # Normalize transition positions
        norm_positions = (row_transitions + 1) / width
        transitions.extend(norm_positions)
    
    # Top to bottom transitions
    for j in range(width):
        col_transitions = np.where(np.diff(binary_image[:, j]) != 0)[0]
        # Normalize transition positions
        norm_positions = (col_transitions + 1) / height
        transitions.extend(norm_positions)
    
    # Bottom to top transitions
    for j in range(width):
        col_flipped = np.flip(binary_image[:, j])
        col_transitions = np.where(np.diff(col_flipped) != 0)[0]
        # Normalize transition positions
        norm_positions = (col_transitions + 1) / height
        transitions.extend(norm_positions)
    
    # Average the transitions to get a feature vector of desired size
    if len(transitions) == 0:
        # No transitions found
        return np.zeros(num_features)
    
    # Sort transitions
    transitions.sort()
    
    # Use averaging to get a fixed-length feature vector
    if len(transitions) <= num_features:
        # Pad with zeros if we have fewer transitions than desired features
        padded = np.zeros(num_features)
        padded[:len(transitions)] = transitions
        return padded
    else:
        # Average transitions to get desired feature length
        indices = np.linspace(0, len(transitions)-1, num_features, dtype=int)
        return np.array([transitions[i] for i in indices])

def extract_sixfold_surface_feature(binary_image):
    """
    Extract the sixfold surface feature from a binarized signature image.
    
    Parameters:
    -----------
    binary_image : numpy.ndarray
        Binary image where signature pixels are white (255)
        
    Returns:
    --------
    list
        A list of six float values representing signature density in each segment
    """
    # Ensure the image is binary (0 and 1)
    if binary_image.max() > 1:
        binary_image = binary_image / 255.0
    
    # Get image dimensions
    height, width = binary_image.shape
    
    # Calculate the width of each vertical segment
    segment_width = width // 3
    
    # Initialize list to store the six surface area proportions
    sixfold_surface = []
    
    # Process each vertical segment
    for i in range(3):
        # Calculate start and end column indices for the current segment
        start_col = i * segment_width
        end_col = (i + 1) * segment_width if i < 2 else width  # Ensure we include all columns
        
        # Extract the current segment
        segment = binary_image[:, start_col:end_col]
        
        # Find coordinates of signature pixels
        y_coords, x_coords = np.where(segment > 0)
        
        # If there are no signature pixels in this segment, use 0 for both parts
        if len(y_coords) == 0:
            sixfold_surface.extend([0.0, 0.0])
            continue
            
        # Calculate center of gravity (centroid) for this segment
        center_y = int(np.mean(y_coords))
        
        # Divide the segment into upper and lower parts based on center of gravity
        upper_segment = segment[:center_y, :]
        lower_segment = segment[center_y:, :]
        
        # Calculate surface area for upper part
        upper_total_pixels = upper_segment.shape[0] * upper_segment.shape[1]
        upper_signature_pixels = np.sum(upper_segment)
        upper_proportion = upper_signature_pixels / upper_total_pixels if upper_total_pixels > 0 else 0.0
        
        # Calculate surface area for lower part
        lower_total_pixels = lower_segment.shape[0] * lower_segment.shape[1]
        lower_signature_pixels = np.sum(lower_segment)
        lower_proportion = lower_signature_pixels / lower_total_pixels if lower_total_pixels > 0 else 0.0
        
        # Add both proportions to our result
        sixfold_surface.extend([upper_proportion, lower_proportion])
    
    return sixfold_surface

def compute_sift(image_path):
    """
    Extract SIFT features from an image.
    
    Parameters:
    -----------
    image_path : str
        Path to the image
        
    Returns:
    --------
    tuple
        (keypoints, descriptors)
    """
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect key points and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

