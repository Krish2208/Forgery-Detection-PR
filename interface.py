import streamlit as st
import os
import numpy as np
import cv2
import pickle
import tempfile
from feature_extraction import extract_signature_features
from train_loop import (
    flatten_features,
    generate_training_examples,
    train_and_evaluate_rf,
    train_and_evaluate_svc,
    train_and_evaluate_rbm
)

# Set page title and configuration
st.set_page_config(page_title="Signature Verification System", layout="wide")

# Title and Description
st.title("Signature Verification System")
st.markdown("""
This application compares a known original signature with a signature of unknown origin
and determines if it's genuine or a forgery.
""")

# Create columns for the two image uploads
col1, col2 = st.columns(2)

# First column for the original signature
with col1:
    st.subheader("Original Signature")
    original_image = st.file_uploader("Upload an original signature", type=["png", "jpg", "jpeg"], key="original")
    
    if original_image is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(original_image.getvalue())
            original_path = temp_file.name
        
        # Display the image
        st.image(original_image, caption="Original Signature", use_column_width=True)

# Second column for the signature to verify
with col2:
    st.subheader("Signature to Verify")
    verify_image = st.file_uploader("Upload a signature to verify", type=["png", "jpg", "jpeg"], key="verify")
    
    if verify_image is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(verify_image.getvalue())
            verify_path = temp_file.name
        
        # Display the image
        st.image(verify_image, caption="Signature to Verify", use_column_width=True)

# Feature selection and model options
st.sidebar.title("Configuration Options")

st.sidebar.subheader("Model Selection")
model_option = st.sidebar.selectbox(
    "Choose a model for verification:",
    ["Random Forest", "Support Vector Machine", "Restricted Boltzmann Machine"]
)

st.sidebar.subheader("Feature Selection")
feature_options = {
    "convexity": st.sidebar.checkbox("Convexity", value=True),
    "signature_density": st.sidebar.checkbox("Signature Density", value=True),
    "compactness": st.sidebar.checkbox("Compactness", value=True),
    "eccentricity": st.sidebar.checkbox("Eccentricity", value=True),
    "hu_moments": st.sidebar.checkbox("Hu Moments", value=True),
    "polar_features": st.sidebar.checkbox("Polar Features", value=True),
    "top_heights": st.sidebar.checkbox("Top Heights", value=True),
    "bottom_heights": st.sidebar.checkbox("Bottom Heights", value=True),
    "left_widths": st.sidebar.checkbox("Left Widths", value=True),
    "right_widths": st.sidebar.checkbox("Right Widths", value=True),
    "sixfold_surface": st.sidebar.checkbox("Sixfold Surface", value=True),
    "transition_features": st.sidebar.checkbox("Transition Features", value=True),
    "mean_sift_descriptor": st.sidebar.checkbox("SIFT Descriptors", value=True)
}

# Display the selected features
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Selected Features:** {sum(feature_options.values())}/{len(feature_options)}")

# Force retrain option
st.sidebar.markdown("---")
force_retrain = st.sidebar.checkbox("Force model retraining", value=False)
if force_retrain:
    st.sidebar.warning("Model will be retrained with selected features")

# Function to preprocess an image before feature extraction
def preprocess_image(image_path):
    """
    Preprocess the image: convert to grayscale, binarize, crop, and resize.
    Returns the path to the processed image.
    """
    # Read the image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not read image {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Find bounding box by locating extreme black pixels
    rows_with_signature = np.where(np.any(binary > 0, axis=1))[0]
    cols_with_signature = np.where(np.any(binary > 0, axis=0))[0]
    
    if len(rows_with_signature) == 0 or len(cols_with_signature) == 0:
        # No signature pixels found, use the original image
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
    
    # Save the processed image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix="_processed.png") as temp_file:
        processed_path = temp_file.name
        cv2.imwrite(processed_path, cropped_resized)
    
    return processed_path

# Function to filter features based on user selection
def filter_features(features, selected_features):
    """Filter the feature dictionary based on user-selected features."""
    filtered = {}
    for key, value in features.items():
        if key in selected_features and selected_features[key]:
            filtered[key] = value
    return filtered

# Create a hash of feature selections for model naming
def get_feature_hash(feature_dict):
    """Create a short hash based on the enabled features"""
    feature_str = ''.join(['1' if feature_dict[key] else '0' for key in sorted(feature_dict.keys())])
    return feature_str

# Function to get or train a model
def get_or_train_model(model_name, feature_options, force_train=False):
    """Load the model from disk if it exists with the same feature configuration, otherwise train it."""
    # Create a feature hash to use in the filename
    feature_hash = get_feature_hash(feature_options)
    model_path = f"models/{model_name}_model_{feature_hash}.pkl"
    scaler_path = f"models/{model_name}_scaler_{feature_hash}.pkl"
    feature_path = f"models/{model_name}_features_{feature_hash}.pkl"
    
    # Check if we need to train a new model
    need_training = force_train or not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(feature_path)
    
    if not need_training:
        # Check if the saved feature configuration matches the current one
        try:
            with open(feature_path, 'rb') as f:
                saved_features = pickle.load(f)
            # If feature configurations don't match, we need to retrain
            if saved_features != feature_options:
                need_training = True
        except:
            # If there's any error loading the feature configuration, retrain
            need_training = True
    
    if need_training:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Create directories for signature data if they don't exist
        for directory in ["signatures/features_org", "signatures/features_forg"]:
            os.makedirs(directory, exist_ok=True)
        
        # Train a new model
        with st.spinner(f"Training {model_name} model on 10,000 examples with selected features... This may take a while."):
            dir1 = "signatures/features_org"
            dir2 = "signatures/features_forg"
            
            # Generate training examples
            examples, labels = generate_training_examples(dir1, dir2, seed=42, num_examples=10000)
            expected_features = [k for k, v in feature_options.items() if v]
            
            # Train the selected model
            if model_name == "rf":
                results = train_and_evaluate_rf(examples, labels, expected_keys=expected_features)
            elif model_name == "svc":
                results = train_and_evaluate_svc(examples, labels, expected_keys=expected_features)
            elif model_name == "rbm":
                results = train_and_evaluate_rbm(examples, labels, expected_keys=expected_features)
            
            # Save the model, scaler, and feature configuration
            with open(model_path, 'wb') as f:
                pickle.dump(results['model'], f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(results['scaler'], f)
            
            return results['model'], results['scaler']
    else:
        # Load existing model and scaler
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler

# Function to make a prediction
def predict_forgery(original_features, verify_features, model, scaler, feature_options):
    """Use the model to predict if the second signature is a forgery."""
    # Filter features based on user selection
    expected_features = [k for k, v in feature_options.items() if v]
    
    # Create feature vectors (difference between the two)
    vector1 = flatten_features(original_features, expected_features)
    vector2 = flatten_features(verify_features, expected_features)
    diff_vector = vector1 - vector2
    
    # Scale the feature vector
    scaled_vector = scaler.transform([diff_vector])
    
    # Make prediction
    prediction = model.predict(scaled_vector)[0]
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled_vector)[0]
    else:
        # For SVC without probability=True
        probabilities = [0.5, 0.5]  # Default if not available
    
    return prediction, probabilities

# Button to verify signatures
if st.button("Verify Signatures", disabled=(original_image is None or verify_image is None)):
    try:
        # Check if images are uploaded
        if original_image is None or verify_image is None:
            st.error("Please upload both an original signature and a signature to verify.")
        else:
            # Preprocess the images
            with st.spinner("Preprocessing images..."):
                processed_original_path = preprocess_image(original_path)
                processed_verify_path = preprocess_image(verify_path)
            
            # Extract features
            with st.spinner("Extracting features..."):
                original_features = extract_signature_features(processed_original_path)
                verify_features = extract_signature_features(processed_verify_path)
            
            # Get the appropriate model
            model_name_map = {
                "Random Forest": "rf",
                "Support Vector Machine": "svc",
                "Restricted Boltzmann Machine": "rbm"
            }
            model_name = model_name_map[model_option]
            
            with st.spinner(f"Loading {model_option} model..."):
                model, scaler = get_or_train_model(model_name, feature_options, force_train=force_retrain)
            
            # Make prediction
            with st.spinner("Comparing signatures..."):
                prediction, probabilities = predict_forgery(
                    original_features, verify_features, model, scaler, feature_options
                )
            
            # Display results
            st.markdown("---")
            st.header("Verification Result")
            
            if prediction == 0:
                st.success("✅ GENUINE - The signatures appear to match.")
                confidence = probabilities[0] if len(probabilities) > 1 else 0.5
            else:
                st.error("❌ FORGERY - The signatures do not appear to match.")
                confidence = probabilities[1] if len(probabilities) > 1 else 0.5
            
            st.markdown(f"**Confidence level:** {confidence:.2%}")
            
            # Display probability chart
            if hasattr(model, 'predict_proba'):
                st.subheader("Probability Distribution")
                
                # Create a bar chart of probabilities
                prob_data = np.array([[probabilities[0], probabilities[1]]])
                st.bar_chart(
                    data=prob_data.T,
                    width=0,
                    height=0,
                    use_container_width=True
                )
                st.markdown("*0 = Genuine, 1 = Forgery*")
            
            # Cleanup temporary files
            for path in [original_path, verify_path, processed_original_path, processed_verify_path]:
                try:
                    os.unlink(path)
                except:
                    pass
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# Add information about the system
with st.expander("About the Signature Verification System"):
    st.markdown("""
    ### How it works
    
    This signature verification system uses machine learning to compare features extracted 
    from signature images. It determines if a signature to verify matches an original signature.
    
    1. **Preprocessing**: Images are converted to grayscale, binarized, cropped to the signature area, and resized.
    2. **Feature Extraction**: Various features are extracted from both images, including shape descriptors, density metrics, and more.
    3. **Feature Comparison**: The difference between features from both signatures is analyzed.
    4. **Classification**: A machine learning model evaluates these differences to determine if the signature is genuine or a forgery.
    
    ### Available Models
    
    - **Random Forest**: Good general-purpose model with high accuracy and resistance to overfitting.
    - **Support Vector Machine**: Strong for finding complex decision boundaries between genuine and forged signatures.
    - **Restricted Boltzmann Machine**: Deep learning approach that can detect subtle patterns in signature differences.
    
    ### Features
    
    The system extracts multiple features from each signature:
    
    - **Convexity**: Ratio of the signature area to its convex hull area
    - **Signature Density**: Density of black pixels in the signature
    - **Compactness**: Measure of how compact the signature is
    - **Eccentricity**: Measure of elongation of the signature
    - **Hu Moments**: Shape descriptors invariant to rotation, scale, and translation
    - **Polar Features**: Features based on the polar representation of the signature
    - **Heights and Widths**: Measurements of the signature's vertical and horizontal components
    - **Sixfold Surface**: Surface measurements in six regions of the signature
    - **Transition Features**: Patterns of transitions between black and white pixels
    - **SIFT Descriptors**: Scale-Invariant Feature Transform descriptors capturing local image features
    """)

# Cache cleanup upon app closure
def cleanup_temp_files():
    """Clean up any temporary files when the application closes."""
    folder = tempfile.gettempdir()
    for filename in os.listdir(folder):
        if filename.endswith("_processed.png"):
            os.unlink(os.path.join(folder, filename))

# Register the cleanup function
import atexit
atexit.register(cleanup_temp_files)