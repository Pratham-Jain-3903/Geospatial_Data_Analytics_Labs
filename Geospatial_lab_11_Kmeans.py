import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage.transform import resize

def load_tiff_file(file_path):
    """Load the .tiff file and return its content."""
    with rasterio.open(file_path) as src:
        data = src.read()  # Read all bands into a 3D array (bands x height x width)
        return data

def spectral_stretch(data):
    """Apply a simple linear stretch to the multispectral data."""
    stretched_data = np.zeros_like(data, dtype=np.float32)
    bands, height, width = data.shape
    
    for i in range(bands):
        band = data[i, :, :]
        min_val = np.min(band)
        max_val = np.max(band)
        stretched_data[i, :, :] = (band - min_val) / (max_val - min_val) * 255  # Normalize and scale to 0-255

    return stretched_data.astype(np.uint8)

def downscale_image(data, factor=4):
    """Downscale the image along x and y by the given factor."""
    bands, height, width = data.shape
    downscaled_data = np.zeros((bands, height // factor, width // factor), dtype=np.float32)

    for i in range(bands):
        downscaled_data[i] = resize(data[i], (height // factor, width // factor), anti_aliasing=True)
    
    return downscaled_data

def apply_kmeans(data, n_clusters=5):
    """Apply KMeans clustering on the image data."""
    bands, height, width = data.shape
    reshaped_data = data.reshape(bands, height * width).T  # Reshape for clustering

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(reshaped_data)
    
    clustered = kmeans.labels_.reshape(height, width)

    # Check the distribution of the clusters
    unique, counts = np.unique(clustered, return_counts=True)
    print(f"KMeans label distribution: {dict(zip(unique, counts))}")

    return clustered

def apply_knn(data, pseudo_labels, n_neighbors=3):
    """Apply KNN classifier on the image data using pseudo-labels."""
    bands, height, width = data.shape
    reshaped_data = data.reshape(bands, height * width).T  # Reshape for KNN
    X_train, X_test, y_train, y_test = train_test_split(reshaped_data, pseudo_labels.flatten(), test_size=0.9, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    predictions = knn.predict(reshaped_data).reshape(height, width)
    return predictions

def visualize_results(original_data, kmeans_result, knn_result):
    """Visualize original image, KMeans and KNN results."""
    rgb_image = np.transpose(original_data[[2, 1, 0], :, :], (1, 2, 0))
    rgb_image = (rgb_image / rgb_image.max() * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original Image
    axes[0].imshow(rgb_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # KMeans result
    axes[1].imshow(kmeans_result, cmap='tab10')
    axes[1].set_title('KMeans Result')
    axes[1].axis('off')

    # KNN result
    axes[2].imshow(knn_result, cmap='tab10')
    axes[2].set_title('KNN Result')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def main(file_path):
    """Main function to load data, downscale it, and apply KMeans and KNN."""
    data = load_tiff_file(file_path)

    # Print the shape of the data for confirmation
    print(f"Original Data shape: {data.shape}")  # (bands, height, width)

    # Spectral stretching
    stretched_data = spectral_stretch(data)

    # Downscale the image by 4 times along x, y
    downscaled_data = downscale_image(stretched_data, factor=4)
    print(f"Downscaled Data shape: {downscaled_data.shape}")

    # Apply KMeans on the downscaled image
    kmeans_result = apply_kmeans(downscaled_data)

    # Use KMeans result as pseudo-labels for KNN
    knn_result = apply_knn(downscaled_data, kmeans_result)

    # Visualize results
    visualize_results(downscaled_data, kmeans_result, knn_result)

if __name__ == '__main__':
    # Path to your .tiff file
    file_path = r"C:\Users\Pratham Jain\SisterDear\Geospatial\raster\Landsat_ETM_2001-08-26_multispectral.tif"
    main(file_path)
