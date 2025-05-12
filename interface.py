import streamlit as st
import os
import numpy as np
import cv2
import pickle
import tempfile
import datetime
import glob
import re
from feature_extraction import extract_signature_features
from train_loop import (
    flatten_features,
    generate_training_examples,
    train_and_evaluate_rf,
    train_and_evaluate_svc,
    train_and_evaluate_rbm
)

feature_sizes = {
    "convexity": 1,
    "signature_density": 1,
    "compactness": 1,
    "eccentricity": 1,
    "hu_moments": 7,
    "polar_features": 60,
    "top_heights": 20,
    "bottom_heights": 20,
    "left_widths": 20,
    "right_widths": 20,
    "sixfold_surface": 6,
    "transition_features": 40,
    "mean_sift_descriptor": 128,
    "lbp_features": 26,
    "hog_features": 3780
}

# Set page title and configuration
st.set_page_config(page_title="Signature Verification System", layout="wide")

# Title and Description
st.title("Signature Verification System")
st.markdown("""
This application compares a known original signature with a signature of unknown origin
and determines if it's genuine or a forgery.
""")

# Sidebar with tabs for different functionalities
st.sidebar.title("Configuration")
sidebar_tab = st.sidebar.radio("Select Mode", ["Signature Verification", "Model Management"])

if sidebar_tab == "Signature Verification":
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

    # Function to list all available trained models
    def list_available_models():
        """List all available model files and extract their metadata."""
        models = []
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Get all model files
        model_files = glob.glob("models/*.pkl")
        
        # Extract metadata from filenames
        for model_file in model_files:
            if "_model_" in model_file:
                base_name = os.path.basename(model_file)
                # Extract model type and feature hash
                match = re.match(r"(\w+)_model_([01]+)(?:_pca_(\d+))?(?:_(\d{14}))?\.pkl", base_name)
                
                if match:
                    model_type, feature_hash, pca_components, timestamp = match.groups()
                    
                    # Map model type to user-friendly name
                    model_name_map = {
                        "rf": "Random Forest",
                        "svc": "Support Vector Machine", 
                        "rbm": "Restricted Boltzmann Machine"
                    }
                    
                    model_name = model_name_map.get(model_type, model_type)
                    
                    # Format timestamp if available
                    if timestamp:
                        formatted_time = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}"
                    else:
                        formatted_time = "Unknown"
                    
                    # Count active features
                    active_features = feature_hash.count('1')
                    total_features = len(feature_hash)
                    
                    # Add PCA info if available
                    pca_info = f" with PCA ({pca_components} components)" if pca_components else ""
                    
                    # Create a display name for the model
                    display_name = f"{model_name} - {active_features}/{total_features} features{pca_info} - {formatted_time}"
                    
                    models.append({
                        "file": base_name,
                        "type": model_type,
                        "feature_hash": feature_hash,
                        "pca_components": pca_components,
                        "timestamp": timestamp,
                        "display_name": display_name
                    })
        
        return models

    # List available models
    available_models = list_available_models()

    # Option to use a previously trained model or configure a new one
    st.sidebar.subheader("Model Selection")
    
    model_option_type = st.sidebar.radio(
        "Choose model option:",
        ["Use existing model", "Configure new model"]
    )
    
    # Model selection and feature configuration
    selected_model_info = None
    model_option = None
    feature_options = {}
    use_pca = False
    pca_components = 0
    force_retrain = False
    
    if model_option_type == "Use existing model":
        if not available_models:
            st.sidebar.warning("No trained models found. Please configure a new model.")
            model_option_type = "Configure new model"
        else:
            model_options = [model["display_name"] for model in available_models]
            selected_model_idx = st.sidebar.selectbox(
                "Select a trained model:",
                range(len(model_options)),
                format_func=lambda i: model_options[i]
            )
            
            selected_model_info = available_models[selected_model_idx]
            st.sidebar.info(f"Using {selected_model_info['display_name']}")
            
            # Display feature information for the selected model
            if selected_model_info:
                feature_hash = selected_model_info['feature_hash']
                model_type = selected_model_info['type']
                
                # Map all possible features
                all_features = [
                    "convexity", "signature_density", "compactness", "eccentricity", 
                    "hu_moments", "polar_features", "top_heights", "bottom_heights", 
                    "left_widths", "right_widths", "sixfold_surface", "transition_features", 
                    "mean_sift_descriptor", "lbp_features", "hog_features"
                ]
                
                # Create feature_options dictionary based on feature_hash
                feature_options = {}
                for i, feature in enumerate(all_features):
                    if i < len(feature_hash):
                        feature_options[feature] = (feature_hash[i] == '1')
                    else:
                        feature_options[feature] = False
                
                # Display features in use
                with st.sidebar.expander("Features in use"):
                    for feature, enabled in feature_options.items():
                        st.markdown(f"- {'✅' if enabled else '❌'} {feature}")
                
                # Set model_option based on model_type
                model_name_map = {
                    "rf": "Random Forest",
                    "svc": "Support Vector Machine",
                    "rbm": "Restricted Boltzmann Machine"
                }
                model_option = model_name_map.get(model_type, "Random Forest")
                
                # Show PCA information if applicable
                if selected_model_info['pca_components']:
                    st.sidebar.info(f"PCA applied with {selected_model_info['pca_components']} components")
                    use_pca = True
                    pca_components = int(selected_model_info['pca_components'])
    
    if model_option_type == "Configure new model":
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
            "mean_sift_descriptor": st.sidebar.checkbox("SIFT Descriptors", value=True),
            "lbp_features": st.sidebar.checkbox("LBP Features", value=True),
            "hog_features": st.sidebar.checkbox("Histogram of Oriented Gradients", value=True)
        }
        
        # Display the selected features
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Selected Features:** {sum(feature_options.values())}/{len(feature_options)}")
        
        # PCA option
        st.sidebar.subheader("Dimensionality Reduction")
        use_pca = st.sidebar.checkbox("Apply PCA", value=False)
        
        if use_pca:
            if sum(feature_options.values()) > 2:
                max_value = 0
                for feature, size in feature_sizes.items():
                    if feature_options[feature]:
                        max_value += size
                pca_components = st.sidebar.number_input(
                    "Number of PCA components",
                    min_value=2,
                    max_value=max_value,
                    value=min(50, sum(feature_options.values())),
                    step=1
                )
        
        # Force retrain option
        st.sidebar.markdown("---")
        force_retrain = st.sidebar.checkbox("Force model retraining", value=False)
        if force_retrain:
            st.sidebar.warning("Model will be retrained with selected features")

elif sidebar_tab == "Model Management":
    st.header("Model Management")
    
    # List all trained models
    def list_available_models_detailed():
        """List all available model files with detailed metadata."""
        models = []
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Get all model files
        model_files = glob.glob("models/*.pkl")
        
        # Extract metadata from filenames
        for model_file in model_files:
            if "_model_" in model_file:
                base_name = os.path.basename(model_file)
                # Extract model type and feature hash
                match = re.match(r"(\w+)_model_([01]+)(?:_pca_(\d+))?(?:_(\d{14}))?\.pkl", base_name)
                
                if match:
                    model_type, feature_hash, pca_components, timestamp = match.groups()
                    
                    # Map model type to user-friendly name
                    model_name_map = {
                        "rf": "Random Forest",
                        "svc": "Support Vector Machine", 
                        "rbm": "Restricted Boltzmann Machine"
                    }
                    
                    model_name = model_name_map.get(model_type, model_type)
                    
                    # Format timestamp if available
                    if timestamp:
                        formatted_time = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}"
                    else:
                        formatted_time = "Unknown"
                    
                    # Map feature hash to features
                    all_features = [
                        "convexity", "signature_density", "compactness", "eccentricity", 
                        "hu_moments", "polar_features", "top_heights", "bottom_heights", 
                        "left_widths", "right_widths", "sixfold_surface", "transition_features", 
                        "mean_sift_descriptor", "lbp_features", "hog_features"
                    ]
                    
                    active_features = []
                    for i, feature in enumerate(all_features):
                        if i < len(feature_hash) and feature_hash[i] == '1':
                            active_features.append(feature)
                    
                    # Add PCA info if available
                    pca_info = f"{pca_components} components" if pca_components else "Not used"
                    
                    # File sizes
                    model_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
                    
                    # Check if scaler file exists and get size
                    scaler_file = model_file.replace("_model_", "_scaler_")
                    if os.path.exists(scaler_file):
                        scaler_size = os.path.getsize(scaler_file) / (1024 * 1024)  # MB
                    else:
                        scaler_size = 0
                    
                    # Check if PCA file exists and get size
                    pca_file = model_file.replace("_model_", "_pca_")
                    if pca_components and os.path.exists(pca_file):
                        pca_size = os.path.getsize(pca_file) / (1024 * 1024)  # MB
                    else:
                        pca_size = 0
                    
                    total_size = model_size + scaler_size + pca_size
                    
                    models.append({
                        "file": base_name,
                        "type": model_type,
                        "model_name": model_name,
                        "feature_hash": feature_hash,
                        "active_features": active_features,
                        "pca_components": pca_components,
                        "pca_info": pca_info,
                        "timestamp": timestamp,
                        "formatted_time": formatted_time,
                        "model_size": model_size,
                        "scaler_size": scaler_size,
                        "pca_size": pca_size,
                        "total_size": total_size
                    })
        
        return models
    
    models = list_available_models_detailed()
    
    if not models:
        st.info("No trained models found. Train a model in the Signature Verification tab.")
    else:
        st.write(f"Found {len(models)} trained models.")
        
        # Create a table of models
        model_data = []
        for model in models:
            model_data.append({
                "Model Type": model["model_name"],
                "Features": f"{len(model['active_features'])}/{len(model['feature_hash'])}",
                "PCA": model["pca_info"],
                "Created": model["formatted_time"],
                "Size": f"{model['total_size']:.2f} MB"
            })
        
        st.table(model_data)
        
        # Option to delete models
        st.subheader("Delete Models")
        
        model_to_delete = st.selectbox(
            "Select model to delete:",
            [f"{m['model_name']} - {len(m['active_features'])}/{len(m['feature_hash'])} features - {m['formatted_time']}" for m in models],
            index=0
        )
        
        if st.button("Delete Selected Model"):
            selected_index = [f"{m['model_name']} - {len(m['active_features'])}/{len(m['feature_hash'])} features - {m['formatted_time']}" for m in models].index(model_to_delete)
            selected_model = models[selected_index]
            
            # Delete model file
            model_file = os.path.join("models", selected_model["file"])
            if os.path.exists(model_file):
                os.remove(model_file)
            
            # Delete scaler file
            scaler_file = model_file.replace("_model_", "_scaler_")
            if os.path.exists(scaler_file):
                os.remove(scaler_file)
            
            # Delete PCA file
            pca_file = model_file.replace("_model_", "_pca_")
            if os.path.exists(pca_file):
                os.remove(pca_file)
            
            st.success(f"Deleted model: {model_to_delete}")
            st.experimental_rerun()
        
        # Model details
        st.subheader("Model Details")
        
        model_to_view = st.selectbox(
            "Select model to view details:",
            [f"{m['model_name']} - {len(m['active_features'])}/{len(m['feature_hash'])} features - {m['formatted_time']}" for m in models],
            index=0
        )
        
        selected_index = [f"{m['model_name']} - {len(m['active_features'])}/{len(m['feature_hash'])} features - {m['formatted_time']}" for m in models].index(model_to_view)
        selected_model = models[selected_index]
        
        # Display model details
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model Information**")
            st.markdown(f"- **Type:** {selected_model['model_name']}")
            st.markdown(f"- **Created:** {selected_model['formatted_time']}")
            st.markdown(f"- **PCA:** {selected_model['pca_info']}")
            st.markdown(f"- **File Sizes:**")
            st.markdown(f"  - Model: {selected_model['model_size']:.2f} MB")
            st.markdown(f"  - Scaler: {selected_model['scaler_size']:.2f} MB")
            if selected_model['pca_components']:
                st.markdown(f"  - PCA: {selected_model['pca_size']:.2f} MB")
            st.markdown(f"  - **Total:** {selected_model['total_size']:.2f} MB")
        
        with col2:
            st.markdown("**Features Used**")
            for feature in selected_model['active_features']:
                st.markdown(f"- {feature}")
else:
    # Code for any other sidebar tabs would go here
    pass

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
def get_or_train_model(model_name, feature_options, use_pca=False, pca_components=10, force_train=False, selected_model_info=None):
    """Load the model from disk if it exists with the same feature configuration, otherwise train it."""
    if use_pca:
        temp_pca_components = 0
        for feature, enabled in feature_options.items():
            if enabled:
                temp_pca_components += feature_sizes[feature]
        if temp_pca_components < pca_components:
            use_pca = False    
    
    # Create a feature hash to use in the filename
    feature_hash = get_feature_hash(feature_options)
    
    # Generate timestamp for new models
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # If we're using a specific model selected by the user
    if selected_model_info is not None:
        model_path = f"models/{selected_model_info['type']}_model_{selected_model_info['feature_hash']}"
        if selected_model_info['pca_components']:
            model_path += f"_pca_{selected_model_info['pca_components']}"
        if selected_model_info['timestamp']:
            model_path += f"_{selected_model_info['timestamp']}"
        model_path += ".pkl"
        
        scaler_path = model_path.replace("_model_", "_scaler_")
        pca_path = model_path.replace("_model_", "_pca_")
        
        # Load existing model and scaler
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load PCA if applicable
        pca = None
        if selected_model_info['pca_components'] and os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
        
        return model, scaler, pca
    
    # For new models or retraining
    # Create a base path for the model without timestamp (for checking if it exists)
    base_path = f"models/{model_name}_model_{feature_hash}"
    if use_pca:
        base_path += f"_pca_{pca_components}"
    
    # Path for new model with timestamp
    new_model_path = f"{base_path}_{timestamp}.pkl"
    new_scaler_path = new_model_path.replace("_model_", "_scaler_")
    new_pca_path = new_model_path.replace("_model_", "_pca_")
    
    # Check if we need to train a new model
    existing_models = glob.glob(f"{base_path}*.pkl")
    need_training = force_train or not existing_models
    
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
                results = train_and_evaluate_rf(examples, labels, expected_keys=expected_features, use_pca=use_pca, pca_components=pca_components)
            elif model_name == "svc":
                results = train_and_evaluate_svc(examples, labels, expected_keys=expected_features, use_pca=use_pca, pca_components=pca_components)
            elif model_name == "rbm":
                results = train_and_evaluate_rbm(examples, labels, expected_keys=expected_features, use_pca=use_pca, pca_components=pca_components)
            
            # Save the model, scaler, and feature configuration
            with open(new_model_path, 'wb') as f:
                pickle.dump(results['model'], f)
            
            with open(new_scaler_path, 'wb') as f:
                pickle.dump(results['scaler'], f)
                
            with open(new_pca_path, 'wb') as f:
                if use_pca:
                    pickle.dump(results['pca'], f)
            
            return results['model'], results['scaler'], results['pca']
    else:
        # Load existing model with the most recent timestamp
        latest_model = max(existing_models)
        latest_scaler = latest_model.replace("_model_", "_scaler_")
        latest_pca = latest_model.replace("_model_", "_pca_")
        
        with open(latest_model, 'rb') as f:
            model = pickle.load(f)
        
        with open(latest_scaler, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load PCA if applicable
        pca = None
        if os.path.exists(latest_pca):
            with open(latest_pca, 'rb') as f:
                pca = pickle.load(f)
        
        return model, scaler, pca

# Function to make a prediction
def predict_forgery(original_features, verify_features, model, scaler, feature_options, pca=None):
    """Use the model to predict if the second signature is a forgery."""
    # Filter features based on user selection
    expected_features = [k for k, v in feature_options.items() if v]
    
    # Create feature vectors (difference between the two)
    vector1 = flatten_features(original_features, expected_features)
    vector2 = flatten_features(verify_features, expected_features)
    diff_vector = vector1 - vector2
    
    # Scale the feature vector
    scaled_vector = scaler.transform([diff_vector])
    
    # Apply PCA if provided
    if pca is not None:
        scaled_vector = pca.transform(scaled_vector)
    
    # Make prediction
    prediction = model.predict(scaled_vector)[0]
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled_vector)[0]
    else:
        # For SVC without probability=True
        probabilities = [0.5, 0.5]  # Default if not available
    
    return prediction, probabilities

# Button to verify signatures in the Signature Verification tab
if sidebar_tab == "Signature Verification" and st.button("Verify Signatures", disabled=(original_image is None or verify_image is None)):
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
                model, scaler, pca = get_or_train_model(
                    model_name, 
                    feature_options, 
                    use_pca=use_pca, 
                    pca_components=pca_components, 
                    force_train=force_retrain,
                    selected_model_info=selected_model_info
                )
            
            # Make prediction
            with st.spinner("Comparing signatures..."):
                prediction, probabilities = predict_forgery(
                    original_features, verify_features, model, scaler, feature_options, pca
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
            
            # Display information about the model used
            st.subheader("Model Information")
            if selected_model_info:
                st.info(f"Used model: {selected_model_info['display_name']}")
            else:
                model_desc = f"{model_option} with {sum(feature_options.values())}/{len(feature_options)} features"
                if use_pca:
                    model_desc += f" and PCA ({pca_components} components)"
                st.info(f"Used model: {model_desc}")
            
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
    - **LBP Features**: Local Binary Patterns capturing texture information
    - **Histogram of Oriented Gradients**: Edge and gradient structure of the signature
    
    ### PCA (Principal Component Analysis)
    
    PCA is a technique that reduces the dimensionality of the feature space while preserving most of the variance.
    Benefits include:
    
    - **Faster Processing**: Reduces computation time by working with fewer dimensions
    - **Noise Reduction**: Can filter out noise in the data
    - **Better Generalization**: May improve performance by removing redundant features
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