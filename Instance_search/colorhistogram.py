import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def compute_histogram(image, bbox=None):
    """Compute a normalized color histogram for the given image and optional bounding box."""
    if bbox:
        x, y, width, height = bbox
        cropped_instance = image[y:y + height, x:x + width]  # Crop region using bbox
    else:
        cropped_instance = image  # Use the full image if no bounding box
    histogram = cv2.calcHist([cropped_instance], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

def read_bbox(file_path):
    """Read the bounding box information from a text file."""
    with open(file_path, 'r') as file:
        bbox = list(map(int, file.readline().strip().split()))
    return bbox

def cosine_similarity(vec1, vec2):
    """Calculate Cosine Similarity between two feature vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def visualize_results(query_image, top_results, gallery_dir):
    """
    Visualize the query image and its top 10 matching gallery images.

    Args:
        query_image (str): Path to the query image.
        top_results (list): List of tuples (gallery_image_name, similarity_score).
        gallery_dir (str): Directory containing gallery images.
    """
    plt.figure(figsize=(15, 5))

    # Display query image
    plt.subplot(2, 6, 1)
    plt.title('Query')
    query_img = cv2.imread(query_image)
    plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Display top 10 retrieved gallery images
    for i, (gallery_name, score) in enumerate(top_results[:10]):
        gallery_path = os.path.join(gallery_dir, gallery_name)
        gallery_img = cv2.imread(gallery_path)
        plt.subplot(2, 6, i + 2)
        plt.title(f"Score: {score:.4f}")
        plt.imshow(cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Paths
download_path = '/content/drive/MyDrive/4186'
path_query = download_path + '/query_img_4186'
path_query_txt = download_path + '/query_img_box_4186'
path_gallery = download_path + '/gallery_4186'

# Query image numbers in the specified order
query_image_numbers = [27, 35, 316, 776, 1258, 1656, 1709, 2032, 2040, 2176,
                       2461, 2714, 3502, 3557, 3833, 3906, 4354, 4445, 4716, 4929]

# Map the query image numbers to their file paths
name_query = [os.path.join(path_query, f"{num}.jpg") for num in query_image_numbers]
name_gallery = glob.glob(path_gallery + '/*.jpg')

# Initialize the rank list
rank_list = []

# Process and visualize for each query image in the specified order
for query_idx, query_path in enumerate(name_query):
    query_name = os.path.basename(query_path)[:-4]
    bbox_path = os.path.join(path_query_txt, f"{query_name}.txt")
    query_bbox = read_bbox(bbox_path)

    # Load query image and compute its histogram
    query_img = cv2.imread(query_path)
    query_hist = compute_histogram(query_img, query_bbox)

    # Compute similarity with all gallery images
    similarities = []
    for gallery_path in name_gallery:
        gallery_img = cv2.imread(gallery_path)
        gallery_hist = compute_histogram(gallery_img)  # Use the entire gallery image
        similarity = cosine_similarity(query_hist, gallery_hist)
        similarities.append((os.path.basename(gallery_path), similarity))

    # Sort gallery images by similarity (descending)
    sorted_gallery = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Append to rank list in the required format
    rank_list.append([gallery_name.split('.')[0] for gallery_name, _ in sorted_gallery])

    # Visualize top 10 results for the current query
    visualize_results(query_path, sorted_gallery, path_gallery)

# Write the rank list to a file
with open('rankList.txt', 'w') as f:
    for query_idx, gallery_rank in enumerate(rank_list):
        f.write(f"Q{query_idx + 1}: {' '.join(gallery_rank)}\n")

print("Ranking completed and saved to rankList.txt.")

