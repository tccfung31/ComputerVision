import numpy as np
import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

def load_gallery_features(gallery_feat_dir):
    """
    Load gallery features for all crops and organize them into a matrix.
    """
    gallery_features = {}
    crop_to_img = []
    all_crops = []
    for feat_file in os.listdir(gallery_feat_dir):
        if feat_file.endswith('.npy'):
            parts = feat_file.split('_crop')
            if len(parts) == 2:
                img_name = parts[0] + '.jpg'
                feat_path = os.path.join(gallery_feat_dir, feat_file)
                feat = np.load(feat_path).flatten()
                if img_name not in gallery_features:
                    gallery_features[img_name] = []
                gallery_features[img_name].append(feat)
                all_crops.append(feat)
                crop_to_img.append(img_name)
    # Filter images with exactly 4 crops
    gallery_files = [img for img, crops in gallery_features.items() if len(crops) == 4]
    gallery_matrix = np.vstack(all_crops)
    return gallery_matrix, crop_to_img, gallery_files

def aggregate_similarities(similarities, crop_to_img, gallery_files, method='max'):
    """
    Aggregate similarities for each gallery image.

    """
    agg_similarities = {img: [] for img in gallery_files}
    for sim, img in zip(similarities, crop_to_img):
        if img in agg_similarities:
            agg_similarities[img].append(sim)
    for img in agg_similarities:
        agg_similarities[img] = max(agg_similarities[img]) if method == 'max' else np.mean(agg_similarities[img])
    return agg_similarities

def retrieval_idx(query_feat_path, gallery_matrix, crop_to_img, gallery_files, agg_method='max'):
    
    #Retrieve the top 10 gallery images most similar to the query based on aggregated crop similarities.

    print(f"Loading query feature from: {query_feat_path}")
    query_feat = np.load(query_feat_path).reshape(1, -1)  # Ensure 2D shape

    # Compute similarities for all crops at once
    similarities = cosine_similarity(query_feat, gallery_matrix).squeeze()

    # Aggregate similarities
    agg_similarities = aggregate_similarities(similarities, crop_to_img, gallery_files, method=agg_method)

    # Get top 10 gallery images
    best_ten = sorted(agg_similarities.items(), key=lambda x: x[1], reverse=True)[:10]
    return best_ten

def visualization(retrieved, query, gallery_img_dir):
    
    #Visualize the query image and top 10 retrieved gallery images.

    plt.figure(figsize=(15, 5))

    # Display query image
    plt.subplot(2, 6, 1)
    plt.title('Query')
    query_img = cv2.imread(query)
    if query_img is None:
        print(f"Error: Could not load query image from {query}")
        return
    plt.imshow(query_img[:, :, ::-1])  # BGR to RGB
    plt.axis('off')

    # Display top 10 retrieved images
    for i, (img_name, sim_score) in enumerate(retrieved):
        img_path = os.path.join(gallery_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load gallery image from {img_path}")
            continue
        plt.subplot(2, 6, i + 2)
        plt.title(f"Sim: {sim_score:.4f}")
        plt.imshow(img[:, :, ::-1])  # BGR to RGB
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Query IDs
    query_ids = [27, 35, 316, 776, 1258, 1656, 1709, 2032, 2040, 2176,
                 2461, 2714, 3502, 3557, 3833, 3906, 4354, 4445, 4716, 4929]

    # Directories
    query_feat_dir = '/content/drive/MyDrive/query_features/'
    query_img_dir = '/content/drive/MyDrive/query_img_4186/'
    gallery_feat_dir = '/content/drive/MyDrive/gallery_features/'
    gallery_img_dir = '/content/drive/MyDrive/gallery_4186/'

    # Load gallery features once
    print("Loading gallery features from:", gallery_feat_dir)
    gallery_matrix, crop_to_img, gallery_files = load_gallery_features(gallery_feat_dir)
    print(f"Loaded features for {len(gallery_files)} gallery images.")

    # Process each query
    for idx, query_id in enumerate(query_ids, 1):
        print(f"\nProcessing query {idx}/{len(query_ids)} - ID: {query_id}")
        query_feat_path = os.path.join(query_feat_dir, f'query_{query_id}.npy')
        query_img_path = os.path.join(query_img_dir, f'{query_id}.jpg')

        # Retrieve and visualize
        best_ten = retrieval_idx(query_feat_path, gallery_matrix, crop_to_img, gallery_files, agg_method='max')
        print(f"Top 10 retrieved images for query {query_id}:")
        for img_name, sim_score in best_ten:
            print(f"{img_name}: {sim_score:.4f}")
        visualization(best_ten, query_img_path, gallery_img_dir)


