import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Initialize global variables for user-defined points
points = {0: [], 1: [], 2: []}
current_cluster = 0

def load_tiff_file(file_path):
    """Load the .tiff file and return its content."""
    with rasterio.open(file_path) as src:
        data = src.read()  # Read all bands into a 3D array (bands x height x width)
        return data

def apply_knn(data, n_neighbors=3):
    """Apply KNN classifier on the image data using selected points."""
    bands, height, width = data.shape
    reshaped_data = data.reshape(bands, height * width).T  # Reshape for KNN

    X_train, y_train = [], []
    for cluster, pts in points.items():
        for (x, y) in pts:
            # Get the feature vector for clicked points
            X_train.append(reshaped_data[y * width + x])  # Correctly index into reshaped data
            y_train.append(cluster)  # Assign the corresponding cluster label
    
    if len(X_train) == 0:  # Check if there are any training points
        raise ValueError("No training points selected for KNN.")

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    predictions = knn.predict(reshaped_data).reshape(height, width)
    return predictions

def visualize_results(original_data, knn_result):
    """Visualize original image and KNN results."""
    rgb_image = np.transpose(original_data[[2, 1, 0], :, :], (1, 2, 0))
    rgb_image = (rgb_image / rgb_image.max() * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original Image
    axes[0].imshow(rgb_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # KNN result
    axes[1].imshow(knn_result, cmap='tab10')
    axes[1].set_title('KNN Result')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def onclick(event):
    """Capture mouse clicks to define points for clustering."""
    global current_cluster
    if event.xdata is not None and event.ydata is not None:
        if len(points[current_cluster]) < 5:  # Limit to 5 points per cluster
            points[current_cluster].append((int(event.xdata), int(event.ydata)))
            print(f"Point added to cluster {current_cluster}: ({int(event.xdata)}, {int(event.ydata)})")
        else:
            print(f"Cluster {current_cluster} already has 5 points.")

def main(file_path):
    """Main function to load data and apply KNN."""
    data = load_tiff_file(file_path)

    print(f"Original Data shape: {data.shape}")  # (bands, height, width)

    # Show original image
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(data[[2, 1, 0], :, :], (1, 2, 0)))  # RGB from bands
    ax.set_title("Click to add points to clusters (0, 1, 2 to switch clusters)")

    # Allow user to switch clusters using keyboard inputs
    def on_key(event):
        global current_cluster
        if event.key in ['0', '1', '2']:
            current_cluster = int(event.key)
            print(f"Switched to cluster {current_cluster}")

    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    
    for _ in range(15):  # Total points = 3 clusters * 5 points
        cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.pause(1)  # Pause to allow clicks
    plt.show()

    # Use KNN with selected points
    knn_result = apply_knn(data)

    visualize_results(data, knn_result)

if __name__ == '__main__':
    file_path = r"C:\Users\Pratham Jain\SisterDear\Geospatial\raster\Landsat_ETM_2001-08-26_multispectral.tif"
    main(file_path)
