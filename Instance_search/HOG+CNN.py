import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # Import gridspec
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, vgg11
from tqdm import tqdm # Optional: for progress bar

# --- Helper Functions (mostly unchanged, added device handling and eval mode) ---

def compute_hog(image, bbox=None, resize_dim=(128, 128)):
    """Compute HOG features for the given image or bounding box."""
    hog = cv2.HOGDescriptor()
    if image is None:
        print("Warning: compute_hog received None image.")
        # Return a zero vector of expected size if possible, otherwise raise error or handle appropriately
        # Example: Determine expected HOG size beforehand or return None/raise error
        # For default HOG parameters and 128x128 input:
        # size = hog.getDescriptorSize() # Requires an image to compute size, tricky.
        # Let's hardcode a typical size or return a clearly identifiable error marker.
        # Hardcoding based on default settings for 128x128: 3780 seems common, but verify.
        # A better approach might be to ensure non-None images are passed.
        # For now, let's return None and handle it later. Or raise an exception.
        # Returning zeros might silently corrupt results if size is wrong.
        # Let's print warning and return None. Calling code must check.
        # raise ValueError("Input image to compute_hog is None") # Safer option
        return None # Requires check in calling code

    try:
        if bbox:
            x, y, width, height = bbox
            # Ensure bbox coords are valid
            y_end = min(y + height, image.shape[0])
            x_end = min(x + width, image.shape[1])
            y = max(y, 0)
            x = max(x, 0)
            if y >= y_end or x >= x_end:
                 print(f"Warning: Invalid bbox {bbox} for image shape {image.shape}. Using full image.")
                 cropped_instance = image
            else:
                 cropped_instance = image[y:y_end, x:x_end]
        else:
            cropped_instance = image

        if cropped_instance.shape[0] == 0 or cropped_instance.shape[1] == 0:
             print(f"Warning: Cropped instance has zero dimension for bbox {bbox}. Using full image.")
             cropped_instance = image # Fallback to full image

        # Resize the image for consistent feature dimensions
        resized_instance = cv2.resize(cropped_instance, resize_dim)
        if len(resized_instance.shape) == 3 and resized_instance.shape[2] == 3:
             gray_image = cv2.cvtColor(resized_instance, cv2.COLOR_BGR2GRAY)
        elif len(resized_instance.shape) == 2:
             gray_image = resized_instance # Already grayscale
        else:
             print(f"Warning: Unexpected image shape {resized_instance.shape} after resize. Trying to convert to gray.")
             # Attempt conversion or handle error
             try:
                 gray_image = cv2.cvtColor(resized_instance, cv2.COLOR_BGR2GRAY)
             except cv2.error as e:
                 print(f"Error converting image to grayscale: {e}. Returning None for HOG.")
                 return None # Requires check


        hog_features = hog.compute(gray_image).flatten()  # Extract and flatten HOG features
        return hog_features
    except Exception as e:
        print(f"Error in compute_hog: {e}")
        return None # Requires check

def read_bbox(file_path):
    """Read the bounding box information from a text file."""
    try:
        with open(file_path, 'r') as file:
            bbox = list(map(int, file.readline().strip().split()))
        # Basic validation: expecting [x, y, width, height] -> 4 numbers
        if len(bbox) != 4:
             print(f"Warning: Bbox file {file_path} does not contain 4 integers. Check format.")
             # Decide on fallback: return None, default bbox, raise error?
             return None # Indicate error
        # Check for non-positive width/height
        if bbox[2] <= 0 or bbox[3] <= 0:
             print(f"Warning: Bbox file {file_path} contains non-positive width/height: {bbox}. Check format.")
             return None # Indicate error

        return bbox
    except FileNotFoundError:
        print(f"Error: Bbox file not found: {file_path}")
        return None # Indicate error
    except ValueError:
        print(f"Error: Could not parse integers in bbox file: {file_path}")
        return None # Indicate error
    except Exception as e:
        print(f"Error reading bbox file {file_path}: {e}")
        return None


def cosine_similarity(vec1, vec2):
    """Calculate Cosine Similarity between two feature vectors."""
    # Ensure vectors are numpy arrays and float type for potentially large numbers
    vec1 = np.asarray(vec1, dtype=np.float64)
    vec2 = np.asarray(vec2, dtype=np.float64)

    # Handle potential zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0  # Or handle as an error case depending on context

    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)

    # Clip similarity to [-1, 1] due to potential floating point inaccuracies
    return np.clip(similarity, -1.0, 1.0)


def extract_cnn_features(image, model, device, bbox=None):
    """Extract CNN features using a pre-trained model."""
    if image is None:
        print("Warning: extract_cnn_features received None image.")
        # Need to return a vector of the correct size (model output size) filled with zeros or handle appropriately.
        # Determine feature size (e.g., 2048 for ResNet50, 4096 for VGG11 adjusted)
        # This is tricky without running the model. Let's hardcode based on known architectures or return None.
        # Returning None is safer as size might vary.
        return None # Requires check

    try:
        if bbox:
            x, y, width, height = bbox
            # Ensure bbox coords are valid within image bounds
            y_end = min(y + height, image.shape[0])
            x_end = min(x + width, image.shape[1])
            y = max(y, 0)
            x = max(x, 0)
            if y >= y_end or x >= x_end:
                 print(f"Warning: Invalid bbox {bbox} for image shape {image.shape}. Using full image.")
                 img_cropped = image
            else:
                 img_cropped = image[y:y_end, x:x_end]
        else:
             img_cropped = image

        if img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
            print(f"Warning: Cropped image has zero dimension for bbox {bbox}. Using full image.")
            img_cropped = image # Fallback


        # Resize and preprocess the image
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img_cropped).unsqueeze(0).to(device) # Add batch dim and move to device

        # Extract features
        with torch.no_grad():
            features = model(input_tensor)
        # Move features back to CPU (if needed) and flatten
        features_np = features.cpu().flatten().numpy()
        return features_np

    except Exception as e:
        print(f"Error in extract_cnn_features: {e}")
        # Optionally log which image/bbox caused the error
        return None # Requires check


def normalize_feature_vector(feature_vector):
    """Normalize a feature vector using L2 norm."""
    if feature_vector is None:
         return None
    norm = np.linalg.norm(feature_vector)
    if norm == 0:
        return feature_vector  # Avoid division by zero, return zero vector
    return feature_vector / norm

# --- Updated Visualization Function ---

def visualize_results(query_image_path, top_results, gallery_dir):
    """
    Visualize the query image and its top 10 matching gallery images
    in the requested format (query top, gallery row below).

    Args:
        query_image_path (str): Path to the query image.
        top_results (list): List of tuples (gallery_image_name, combined_similarity_score).
        gallery_dir (str): Directory containing gallery images.
    """
    num_results_to_show = min(10, len(top_results))
    if num_results_to_show == 0:
        print("No results to visualize.")
        return

    fig = plt.figure(figsize=(18, 5)) # Adjust figsize as needed
    # Create a grid: 2 rows, 10 columns. Query spans top row. Gallery in bottom row.
    gs = gridspec.GridSpec(2, num_results_to_show, height_ratios=[1, 1], wspace=0.1, hspace=0.2)

    # Display query image in the top row, spanning all columns
    ax_query = fig.add_subplot(gs[0, :])
    ax_query.set_title(f'Query: {os.path.basename(query_image_path)}')
    query_img = cv2.imread(query_image_path)
    if query_img is not None:
        ax_query.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    else:
        ax_query.text(0.5, 0.5, 'Query Image Not Found', ha='center', va='center')
    ax_query.axis('off')

    # Display top N retrieved gallery images in the bottom row
    for i in range(num_results_to_show):
        gallery_name, score = top_results[i]
        gallery_path = os.path.join(gallery_dir, gallery_name)
        gallery_img = cv2.imread(gallery_path)

        ax_gallery = fig.add_subplot(gs[1, i])
        ax_gallery.set_title(f"Rank {i+1}\nScore: {score:.3f}", fontsize=8) # Combine Rank and Score

        if gallery_img is not None:
            ax_gallery.imshow(cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB))
        else:
            ax_gallery.text(0.5, 0.5, 'Not Found', ha='center', va='center', fontsize=8)
        ax_gallery.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

# --- Main Execution Logic ---

# Paths (Ensure these are correct for your Google Drive setup)
try:
     from google.colab import drive
     drive.mount('/content/drive')
     drive_mounted = True
except ImportError:
     print("Not running in Google Colab or drive mounting failed. Using relative paths.")
     drive_mounted = False

# Use absolute path if mounted, otherwise assume script is run from a dir containing the data folders
if drive_mounted:
    base_path = '/content/drive/MyDrive/4186'
else:
    base_path = '.' # Assume data folders are in the same directory as the script

path_query = os.path.join(base_path, 'query_img_4186')
path_query_txt = os.path.join(base_path, 'query_img_box_4186')
path_gallery = os.path.join(base_path, 'gallery_4186')

# Check if directories exist
if not os.path.isdir(path_query): raise FileNotFoundError(f"Query image directory not found: {path_query}")
if not os.path.isdir(path_query_txt): raise FileNotFoundError(f"Query bbox directory not found: {path_query_txt}")
if not os.path.isdir(path_gallery): raise FileNotFoundError(f"Gallery image directory not found: {path_gallery}")


# Query image numbers in the specified order
query_image_numbers = [27, 35, 316, 776, 1258, 1656, 1709, 2032, 2040, 2176,
                       2461, 2714, 3502, 3557, 3833, 3906, 4354, 4445, 4716, 4929]

# Map the query image numbers to their file paths and check existence
name_query = []
for num in query_image_numbers:
    q_path = os.path.join(path_query, f"{num}.jpg")
    if not os.path.exists(q_path):
        print(f"Warning: Query image file not found: {q_path}")
        # Decide how to handle missing query files: skip, error out? Skipping for now.
        continue # Skip this query number
    name_query.append(q_path)

name_gallery = glob.glob(os.path.join(path_gallery, '*.jpg'))
if not name_gallery:
    raise FileNotFoundError(f"No gallery images found in {path_gallery}. Check path and file extensions.")

print(f"Found {len(name_query)} query images (after checking existence).")
print(f"Found {len(name_gallery)} gallery images.")


# --- Model Loading and Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained CNN models (ResNet50 and VGG11)
print("Loading ResNet50 model...")
resnet_model = resnet50(pretrained=True)
resnet_model.fc = torch.nn.Identity()  # Remove classification layer
resnet_model = resnet_model.to(device) # Move model to device
resnet_model.eval()                    # Set to evaluation mode

print("Loading VGG11 model...")
vgg_model = vgg11(pretrained=True)
# Adjust VGG feature extraction layer (take output before final classification layer)
# Option 1: Features before classifier
# vgg_model.classifier = torch.nn.Identity() # Gives features from convolutional part only (512*7*7) - large vector!
# Option 2: Features from one layer before the *final* classification layer (e.g., 4096)
num_features_before_last = vgg_model.classifier[-1].in_features # Get input features of the last linear layer
vgg_model.classifier = torch.nn.Sequential(*list(vgg_model.classifier.children())[:-1]) # Remove last layer
vgg_model = vgg_model.to(device) # Move model to device
vgg_model.eval()                 # Set to evaluation mode
# Note: The exact size of VGG features might need tuning depending on which layer you choose.
# The example uses [-1], which removes the *last* Linear layer (usually prediction).
# The size will likely be 4096 for default VGG11 classifier structure.

# --- Feature Extraction and Ranking ---
rank_list = []
combined_feature_cache = {} # Optional: Cache gallery features if memory allows

# Pre-calculate gallery features (optional but recommended for speed)
# Set to False if memory is limited
PRECOMPUTE_GALLERY = True
print(f"Pre-computing gallery features: {PRECOMPUTE_GALLERY}")

if PRECOMPUTE_GALLERY:
    gallery_iterator = tqdm(name_gallery, desc="Precomputing Gallery Features")
    for gallery_path in gallery_iterator:
        gallery_name = os.path.basename(gallery_path)
        gallery_img = cv2.imread(gallery_path)
        if gallery_img is None:
            print(f"Warning: Could not read gallery image {gallery_path}. Skipping.")
            combined_feature_cache[gallery_name] = None # Mark as invalid
            continue

        # Compute features (no bbox for gallery)
        gal_hog_feat = compute_hog(gallery_img, bbox=None, resize_dim=(128, 128))
        gal_res_feat = extract_cnn_features(gallery_img, resnet_model, device=device, bbox=None)
        gal_vgg_feat = extract_cnn_features(gallery_img, vgg_model, device=device, bbox=None)

        # Normalize features
        norm_gal_hog = normalize_feature_vector(gal_hog_feat)
        norm_gal_res = normalize_feature_vector(gal_res_feat)
        norm_gal_vgg = normalize_feature_vector(gal_vgg_feat)

        # Handle potential None returns from feature extraction/normalization
        if norm_gal_hog is None or norm_gal_res is None or norm_gal_vgg is None:
             print(f"Warning: Feature extraction failed for gallery image {gallery_name}. Skipping.")
             combined_feature_cache[gallery_name] = None # Mark as invalid
             continue

        # *** Combine normalized features ***
        gallery_combined_features = np.concatenate((norm_gal_res, norm_gal_vgg, norm_gal_hog))
        # Optional: Normalize the *combined* vector as well? Usually not necessary after combining normalized components for cosine sim.
        # gallery_combined_features = normalize_feature_vector(gallery_combined_features)

        combined_feature_cache[gallery_name] = gallery_combined_features # Store the final combined vector


print("\n--- Processing Queries ---")
# Process and visualize for each query image
query_iterator = tqdm(name_query, desc="Processing Queries")
for query_idx, query_path in enumerate(query_iterator):
    query_img_number = os.path.basename(query_path)[:-4] # Get the number string
    bbox_path = os.path.join(path_query_txt, f"{query_img_number}.txt")

    # Load query image
    query_img = cv2.imread(query_path)
    if query_img is None:
        print(f"\nError: Could not read query image {query_path}. Skipping query {query_img_number}.")
        rank_list.append([]) # Add an empty list for this failed query
        continue

    # Read query bounding box
    query_bbox = read_bbox(bbox_path)
    if query_bbox is None:
        print(f"\nWarning: Could not read or validate bbox for query {query_img_number}. Using full image.")
        # Using full image if bbox fails
        query_bbox = None # Set bbox to None to use full image in feature extraction

    # Compute and normalize query features
    query_hog_features = compute_hog(query_img, query_bbox, resize_dim=(128, 128))
    query_resnet_features = extract_cnn_features(query_img, resnet_model, device=device, bbox=query_bbox)
    query_vgg_features = extract_cnn_features(query_img, vgg_model, device=device, bbox=query_bbox)

    norm_query_hog = normalize_feature_vector(query_hog_features)
    norm_query_res = normalize_feature_vector(query_resnet_features)
    norm_query_vgg = normalize_feature_vector(query_vgg_features)

    # Handle potential None returns
    if norm_query_hog is None or norm_query_res is None or norm_query_vgg is None:
         print(f"\nError: Feature extraction failed for query image {query_img_number}. Skipping query.")
         rank_list.append([])
         continue

    # *** Combine normalized query features ***
    query_combined_features = np.concatenate((norm_query_res, norm_query_vgg, norm_query_hog))
    # Optional: query_combined_features = normalize_feature_vector(query_combined_features)

    # Compute similarity with all gallery images
    similarities = []
    gallery_list_iterator = name_gallery # Iterate through paths if not precomputing
    if PRECOMPUTE_GALLERY:
        gallery_list_iterator = combined_feature_cache.items() # Iterate through cached features

    for item in gallery_list_iterator:
        if PRECOMPUTE_GALLERY:
            gallery_name, gallery_combined_features = item
            if gallery_combined_features is None: # Skip invalid cache entries
                 continue
        else:
            # Compute gallery features on the fly (slower)
            gallery_path = item
            gallery_name = os.path.basename(gallery_path)
            gallery_img = cv2.imread(gallery_path)
            if gallery_img is None: continue # Skip unreadable images

            gal_hog_feat = compute_hog(gallery_img, bbox=None, resize_dim=(128, 128))
            gal_res_feat = extract_cnn_features(gallery_img, resnet_model, device=device, bbox=None)
            gal_vgg_feat = extract_cnn_features(gallery_img, vgg_model, device=device, bbox=None)

            norm_gal_hog = normalize_feature_vector(gal_hog_feat)
            norm_gal_res = normalize_feature_vector(gal_res_feat)
            norm_gal_vgg = normalize_feature_vector(gal_vgg_feat)

            if norm_gal_hog is None or norm_gal_res is None or norm_gal_vgg is None: continue # Skip if features failed

            gallery_combined_features = np.concatenate((norm_gal_res, norm_gal_vgg, norm_gal_hog))
            # Optional: gallery_combined_features = normalize_feature_vector(gallery_combined_features)

        # *** Calculate similarity using combined features ***
        similarity_score = cosine_similarity(query_combined_features, gallery_combined_features)
        similarities.append((gallery_name, similarity_score))

    # Sort gallery images by combined similarity (descending)
    sorted_gallery = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Append to rank list in the required format (just the image IDs/names without extension)
    rank_list.append([gallery_name.split('.')[0] for gallery_name, _ in sorted_gallery])

    # Visualize top 10 results for the current query
    print(f"\nVisualizing results for query: {query_img_number}.jpg")
    visualize_results(query_path, sorted_gallery, path_gallery) # Pass the query path

# --- Save Rank List ---
output_filename = 'rankList_Combined.txt'
print(f"\nSaving ranking list to {output_filename}...")
with open(output_filename, 'w') as f:
    for query_idx, gallery_rank in enumerate(rank_list):
        # Find the original query number corresponding to this index
        # This assumes name_query and rank_list maintain order relative to query_image_numbers
        # Need to handle cases where queries were skipped.
        original_query_num = query_image_numbers[query_idx] # Be careful if queries were skipped! Let's refine this logic.

        # More robust way to get query number if skips occurred:
        # Re-map based on the actual processed queries in name_query
        processed_query_path = name_query[query_idx]
        processed_query_num_str = os.path.basename(processed_query_path)[:-4]

        # The prompt requested Q1, Q2 format... mapping query_idx might be sufficient if no skips.
        # If using Q{idx+1} format is required regardless of actual query number:
        f.write(f"Q{query_idx + 1}: {' '.join(gallery_rank)}\n")
        # If using the actual query number is needed:
        # f.write(f"Q{processed_query_num_str}: {' '.join(gallery_rank)}\n") # Use this if actual numbers matter

print("Ranking completed and saved.")