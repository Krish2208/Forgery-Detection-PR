import os
import random
import re
import json
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from typing import List, Tuple, Dict, Optional, Union

def list_files(directory):
    """List all JSON files in the given directory."""
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return []
    
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            files.append(os.path.join(directory, filename))
    return files

def extract_id(filename):
    """Extract the ID number from the filename."""
    basename = os.path.basename(filename)
    match = re.search(r'(?:forgeries|original)_(\d+)_\d+_features\.json', basename)
    if match:
        return int(match.group(1))
    return None

def group_files_by_id(files):
    """Group files by their ID numbers."""
    grouped = {}
    for file in files:
        id_num = extract_id(file)
        if id_num is not None:
            if id_num not in grouped:
                grouped[id_num] = []
            grouped[id_num].append(file)
    return grouped

def generate_training_examples(dir1, dir2, seed, num_examples=100, load_data=False):
    """Generate training examples and labels.
    
    Args:
        dir1: Directory containing first examples (e.g., signatures/features_forg)
        dir2: Directory containing second examples (e.g., signatures/features_org)
        seed: Random seed for reproducibility
        num_examples: Number of training examples to generate
        load_data: If True, load and return JSON data; if False, return file paths
        
    Returns:
        tuple: (training_examples, labels)
    """
    random.seed(seed)
    
    # List files in both directories
    files_dir1 = list_files(dir1)
    files_dir2 = list_files(dir2)
    
    if not files_dir1:
        print(f"No files found in directory: {dir1}")
        return [], []
    
    # Group files by ID
    grouped_dir1 = group_files_by_id(files_dir1)
    grouped_dir2 = group_files_by_id(files_dir2)
    
    # Find valid IDs (IDs that have multiple files in dir1 or have files in both dir1 and dir2)
    valid_ids = []
    for id_num, files in grouped_dir1.items():
        if len(files) > 1 or id_num in grouped_dir2:
            valid_ids.append(id_num)
    
    if not valid_ids:
        print("No valid IDs found. Each ID must have either multiple files in the first directory or files in both directories.")
        return [], []
    
    training_examples = []
    labels = []
    
    # Try to generate the requested number of examples
    attempts = 0
    max_attempts = num_examples * 10  # Limit the number of attempts to avoid infinite loops
    
    while len(training_examples) < num_examples and attempts < max_attempts:
        attempts += 1
        
        # Randomly select a valid ID
        id_num = random.choice(valid_ids)
        
        # Select first example from dir1
        first_example = random.choice(grouped_dir1[id_num])
        
        # Check if this ID exists in dir2
        has_dir2_examples = id_num in grouped_dir2
        
        # Randomly decide which directory to pick the second example from
        if has_dir2_examples and random.random() < 0.5:
            # Pick from dir2
            second_example = random.choice(grouped_dir2[id_num])
            label = 1  # Different directory label
        else:
            # Pick from dir1 (same directory)
            second_examples = [f for f in grouped_dir1[id_num] if f != first_example]
            if not second_examples:
                continue  # Skip if there's no other example with the same ID
            second_example = random.choice(second_examples)
            label = 0  # Same directory label
        
        if load_data:
            # Load JSON data
            try:
                with open(first_example, 'r') as f1, open(second_example, 'r') as f2:
                    first_data = json.load(f1)
                    second_data = json.load(f2)
                training_examples.append((first_data, second_data))
            except Exception as e:
                print(f"Error loading data: {e}")
                continue
        else:
            # Just store file paths
            training_examples.append((first_example, second_example))
        
        labels.append(label)
    
    if len(training_examples) < num_examples:
        print(f"Warning: Could only generate {len(training_examples)} examples out of {num_examples} requested.")
    
    return training_examples, labels

def list_files_by_id(directory: str) -> Dict[int, List[str]]:
    """
    Lists JSON files in a directory and organizes them by ID.
    
    Args:
        directory: Path to the directory
        
    Returns:
        Dictionary mapping IDs to lists of filenames
    """
    files_by_id = {}
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return files_by_id
    
    for filename in os.listdir(directory):
        if not filename.endswith("_features.json"):
            continue
            
        # Parse ID from filename (e.g., "forgeries_1_2_features.json" -> ID is 1)
        parts = filename.split('_')
        if len(parts) >= 3:
            try:
                id_num = int(parts[1])  # ID is the second part after splitting
                if id_num not in files_by_id:
                    files_by_id[id_num] = []
                files_by_id[id_num].append(os.path.join(directory, filename))
            except ValueError:
                print(f"Warning: Could not parse ID from filename {filename}")
    
    return files_by_id

def generate_training_data(
    dir1: str,
    dir2: str,
    num_examples: Optional[int] = None,
    random_seed: int = 42,
    load_data: bool = False,
    verbose: bool = False
) -> Tuple[List[Union[Tuple[str, str], Tuple[dict, dict]]], List[int]]:
    """
    Generates training examples by pairing files with matching IDs.
    
    Args:
        dir1: First directory (e.g., features_forg)
        dir2: Second directory (e.g., features_org)
        num_examples: Number of examples to generate (if None, generate all possible pairs)
        random_seed: Random seed for reproducibility
        load_data: If True, load the actual JSON data; if False, just return file paths
        verbose: If True, print additional information during processing
        
    Returns:
        Tuple of (examples, labels) where:
        - examples are pairs of (file1, file2) or (data1, data2)
        - labels are 0 if file2 is from dir1, 1 if file2 is from dir2
    """
    random.seed(random_seed)
    
    if verbose:
        print(f"Listing files from directory 1: {dir1}")
    files_by_id_dir1 = list_files_by_id(dir1)
    if verbose:
        print(f"Found {sum(len(files) for files in files_by_id_dir1.values())} files across {len(files_by_id_dir1)} IDs in directory 1")
    
    if verbose:
        print(f"Listing files from directory 2: {dir2}")
    files_by_id_dir2 = list_files_by_id(dir2)
    if verbose:
        print(f"Found {sum(len(files) for files in files_by_id_dir2.values())} files across {len(files_by_id_dir2)} IDs in directory 2")
    
    examples = []
    labels = []
    
    # For each ID in dir1
    for id_num, files1 in files_by_id_dir1.items():
        if verbose:
            print(f"Processing ID {id_num} with {len(files1)} files in directory 1")
        
        for file1 in files1:
            # Option 1: Pick second example from dir1 (same directory)
            for file2 in files1:
                if file1 != file2:  # Don't pair a file with itself
                    if load_data:
                        try:
                            data1 = load_json_data(file1)
                            data2 = load_json_data(file2)
                            examples.append((data1, data2))
                            labels.append(0)  # Label 0: both from dir1
                        except Exception as e:
                            if verbose:
                                print(f"Error loading data from {file1} and {file2}: {e}")
                    else:
                        examples.append((file1, file2))
                        labels.append(0)  # Label 0: both from dir1
            
            # Option 2: Pick second example from dir2 (different directory)
            if id_num in files_by_id_dir2:
                if verbose:
                    print(f"  ID {id_num} also exists in directory 2 with {len(files_by_id_dir2[id_num])} files")
                
                for file2 in files_by_id_dir2[id_num]:
                    if load_data:
                        try:
                            data1 = load_json_data(file1)
                            data2 = load_json_data(file2)
                            examples.append((data1, data2))
                            labels.append(1)  # Label 1: second from dir2
                        except Exception as e:
                            if verbose:
                                print(f"Error loading data from {file1} and {file2}: {e}")
                    else:
                        examples.append((file1, file2))
                        labels.append(1)  # Label 1: second from dir2
    
    if verbose:
        print(f"Generated {len(examples)} examples before sampling")
    
    # Random sampling if num_examples is specified
    if num_examples is not None and num_examples < len(examples):
        if verbose:
            print(f"Randomly selecting {num_examples} examples")
        
        indices = list(range(len(examples)))
        random.shuffle(indices)
        selected_indices = indices[:num_examples]
        
        examples = [examples[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]
    
    return examples, labels

def load_json_data(file_path: str) -> dict:
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_json(file_path):
    """Load JSON file from path."""
    with open(file_path, 'r') as f:
        return json.load(f)

def flatten_features(json_data, expected_keys=None):
    """Flatten all feature values from the JSON data, handling nested structures."""
    
    # Define expected feature keys
    if expected_keys is None:
        # Default expected keys if not provided
        expected_keys = [
            'convexity', 'signature_density', 'compactness', 'eccentricity', 
            'hu_moments', 'polar_features', 'top_heights', 'bottom_heights', 
            'left_widths', 'right_widths', 'sixfold_surface', 
            'transition_features', 'mean_sift_descriptor', 'lbp_features',
            'hog_features'
        ]
    
    # Initialize flattened list
    flattened = []
    
    # Process each key
    for key in expected_keys:
        if key in json_data:
            value = json_data[key]
            
            # Convert the value to a numpy array and flatten it, handling any nesting
            if isinstance(value, (list, tuple)):
                # Handle nested lists by converting to ndarray first
                try:
                    # Try to convert to numpy array and flatten
                    np_array = np.array(value, dtype=float)
                    flattened.extend(np_array.flatten())
                except (ValueError, TypeError):
                    # If conversion fails (e.g., for mixed types or jagged arrays),
                    # process each element recursively
                    for item in value:
                        if isinstance(item, (list, tuple)):
                            # Recursively flatten nested lists
                            flattened.extend(np.array(item, dtype=float).flatten())
                        elif isinstance(item, (int, float)):
                            flattened.append(item)
                        elif isinstance(item, dict):
                            # For dict items in a list, add all values
                            for subvalue in item.values():
                                if isinstance(subvalue, (list, tuple)):
                                    flattened.extend(np.array(subvalue, dtype=float).flatten())
                                else:
                                    flattened.append(float(subvalue) if isinstance(subvalue, (int, float)) else 0)
                        else:
                            flattened.append(0)  # For non-numeric items
            elif isinstance(value, (int, float)):
                flattened.append(value)
            elif isinstance(value, dict):
                # Handle dictionaries by flattening each value
                for subvalue in value.values():
                    if isinstance(subvalue, (list, tuple)):
                        flattened.extend(np.array(subvalue, dtype=float).flatten())
                    elif isinstance(subvalue, (int, float)):
                        flattened.append(subvalue)
                    elif isinstance(subvalue, dict):
                        # Handle nested dictionaries
                        for nested_value in subvalue.values():
                            if isinstance(nested_value, (list, tuple)):
                                flattened.extend(np.array(nested_value, dtype=float).flatten())
                            elif isinstance(nested_value, (int, float)):
                                flattened.append(nested_value)
                            else:
                                flattened.append(0)
                    else:
                        flattened.append(0)
            else:
                # Handle unexpected types by adding as 0
                flattened.append(0)
        else:
            # If key doesn't exist, add placeholder
            flattened.append(0)
    
    return np.array(flattened, dtype=float)

def create_feature_vectors(examples, expected_keys=None):
    """
    Process a list of example pairs to create feature vectors.
    
    Args:
        examples: List of tuples, each containing paths to two JSON files
        
    Returns:
        List of feature vectors (differences between the two JSONs in each pair)
    """
    feature_vectors = []
    
    for json_path1, json_path2 in examples:
        # Load both JSONs
        json1 = load_json(json_path1)
        json2 = load_json(json_path2)
        
        # Flatten each JSON
        vector1 = flatten_features(json1, expected_keys)
        vector2 = flatten_features(json2, expected_keys)
        
        # Create difference vector
        diff_vector = vector1 - vector2
        
        feature_vectors.append(diff_vector)
    
    return np.array(feature_vectors)

def train_and_evaluate_rf(examples, labels, test_size=0.2, random_state=42, expected_keys=None, use_pca=False, pca_components=10):
    """
    Train a Random Forest classifier and evaluate its performance.
    
    Args:
        examples: List of tuples, each containing paths to two JSON files
        labels: List of binary labels (0 or 1)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        expected_keys: List of expected feature keys to extract from JSON
        use_pca: If True, apply PCA for dimensionality reduction
        pca_components: Number of components to keep if PCA is used
        
    Returns:
        Trained model, scaler, test accuracy, and detailed metrics
    """
    # Create feature vectors
    X = create_feature_vectors(examples, expected_keys)
    y = np.array(labels)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optionally apply PCA
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
    else:
        pca = None
    
    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    
    # Return results
    return {
        'model': clf,
        'scaler': scaler,
        'test_accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'cv_scores': cv_scores,
        'feature_importances': clf.feature_importances_,
        'pca': pca
    }



def train_and_evaluate_svc(examples, labels, test_size=0.2, random_state=42, C=1.0, kernel='rbf', gamma='scale', expected_keys=None, use_pca=False, pca_components=10):
    """
    Train a Support Vector Classifier (SVC) and evaluate its performance.
    
    Args:
        examples: List of tuples, each containing paths to two JSON files
        labels: List of binary labels (0 or 1)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        C: Regularization parameter
        kernel: Kernel type to be used in the algorithm
        gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        expected_keys: List of expected feature keys to extract from JSON
        use_pca: If True, apply PCA for dimensionality reduction
        pca_components: Number of components to keep if PCA is used
        
    Returns:
        Trained model, scaler, test accuracy, and detailed metrics
    """
    # Create feature vectors
    X = create_feature_vectors(examples, expected_keys)
    y = np.array(labels)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features - SVC is sensitive to feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optionally apply PCA
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    else:
        pca = None
    
    # Train SVC
    clf = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=random_state)
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    
    # Get decision function values for ROC curve calculation
    y_scores = clf.decision_function(X_test_scaled)
    
    # Return results
    result_dict = {
        'model': clf,
        'scaler': scaler,
        'test_accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'cv_scores': cv_scores,
        'decision_scores': y_scores,
        'pca': pca
    }
        
    return result_dict

def train_and_evaluate_rbm(examples, labels, test_size=0.2, random_state=42, n_components=1024, learning_rate=0.05, n_iter=40, expected_keys=None, use_pca=False, pca_components=10):
    """
    Train a Restricted Boltzmann Machine + Logistic Regression pipeline and evaluate performance.
    
    Args:
        examples: List of tuples, each containing paths to two JSON files
        labels: List of binary labels (0 or 1)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        n_components: Number of hidden units in RBM
        learning_rate: Learning rate for RBM
        n_iter: Number of iterations/epochs for RBM
        expected_keys: List of expected feature keys to extract from JSON
        use_pca: If True, apply PCA for dimensionality reduction
        pca_components: Number of components to keep if PCA is used
        
    Returns:
        Trained model, scaler, test accuracy, and detailed metrics
    """
    # Create feature vectors
    X = create_feature_vectors(examples, expected_keys)
    y = np.array(labels)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optionally apply PCA
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
    else:
        pca = None
    
    # Set up RBM + LogisticRegression pipeline
    # RBM for feature learning + LogisticRegression for classification
    rbm = BernoulliRBM(n_components=n_components, 
                       learning_rate=learning_rate,
                       n_iter=n_iter,
                       random_state=random_state,
                       verbose=True)
    
    logistic = LogisticRegression(random_state=random_state)
    
    classifier = Pipeline(steps=[('rbm', rbm), 
                                 ('logistic', logistic)])
    
    # Train the pipeline
    classifier.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test_scaled)
    
    # Get probability estimates for ROC curve
    y_scores = classifier.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Use cross-validation on the training set
    cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=5)
    
    # Return results
    return {
        'model': classifier,
        'scaler': scaler,
        'test_accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'cv_scores': cv_scores,
        'probability_scores': y_scores,
        'pca': pca
    }

def main():
    dir1 = "signatures/features_org"
    dir2 = "signatures/features_forg"
    
    # Generate training examples with seed 42
    examples, labels = generate_training_examples(dir1, dir2, seed=42, num_examples=10000)
    
    print(f"Generated {len(examples)} training examples")
    print(f"Label distribution: {sum(labels)} examples with label 1 (from dir2), {len(labels) - sum(labels)} with label 0 (from dir1)")
    
    # Train and evaluate model
    print("Training and evaluating model (Random Forest)...")
    results = train_and_evaluate_rf(examples, labels, use_pca=True, pca_components=3)
    
    # Print results
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Cross-validation scores: {results['cv_scores']}")
    print("Classification Report:")
    print(results['classification_report'])
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    
    # Examine feature importance
    print("\nFeature Importances:")
    # Get indices of top 10 most important features
    top_indices = np.argsort(results['feature_importances'])[-200:]
    for i in top_indices:
        print(f"Feature {i}: {results['feature_importances'][i]:.4f}")


    # Train and evaluate model
    print("Training and evaluating model (SVC)...")
    results = train_and_evaluate_svc(examples, labels, use_pca=True, pca_components=2)
    
    # Print results
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Cross-validation scores: {results['cv_scores']}")
    print("Classification Report:")
    print(results['classification_report'])
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    
    # Train and evaluate Boltzmann Machine
    print("Training and evaluating model (Restricted Boltzmann Machine)...")
    results = train_and_evaluate_rbm(examples, labels, use_pca=True, pca_components=2)
    
    # Print results
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Cross-validation scores: {results['cv_scores']}")
    print("Classification Report:")
    print(results['classification_report'])
    print("Confusion Matrix:")
    print(results['confusion_matrix'])

# if __name__ == "__main__":
#     main()