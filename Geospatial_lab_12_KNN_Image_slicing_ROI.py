import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import random

# Initialize global variables for ROI
roi_points = []

def load_tiff_file(file_path):
    """Load the .tiff file and return its content."""
    with rasterio.open(file_path) as src:
        data = src.read()  # Read all bands into a 3D array (bands x height x width)
        return data

def random_sample_points(data, roi, num_samples=100):
    """Randomly sample points from the defined ROI."""
    x_min, x_max = min(roi[0]), max(roi[0])
    y_min, y_max = min(roi[1]), max(roi[1])
    
    # Extract pixel values within the ROI
    roi_pixels = []
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            roi_pixels.append((x, y))
    
    # Randomly select points, ensuring no duplicates
    sampled_points = random.sample(roi_pixels, min(num_samples, len(roi_pixels)))
    
    return sampled_points

def apply_knn(data, sampled_points, n_neighbors=3):
    """Apply KNN classifier on the image data using sampled points."""
    bands, height, width = data.shape
    reshaped_data = data.reshape(bands, height * width).T  # Reshape for KNN

    X_train = [reshaped_data[y * width + x] for (x, y) in sampled_points]  # Get feature vectors
    y_train = [0] * len(sampled_points)  # Assign dummy labels for KNN

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

def onclick_roi(event):
    """Capture mouse clicks to define ROI."""
    global roi_points
    if event.xdata is not None and event.ydata is not None:
        roi_points.append((int(event.xdata), int(event.ydata)))
        print(f"Point added to ROI: ({int(event.xdata)}, {int(event.ydata)})")

def finalize_roi(event):
    """Finalize ROI selection after two clicks."""
    if len(roi_points) == 2:
        print(f"ROI finalized: {roi_points}")
        plt.close()

def main(file_path):
    """Main function to load data and apply KNN."""
    global roi_points
    data = load_tiff_file(file_path)

    print(f"Original Data shape: {data.shape}")  # (bands, height, width)

    # Show original image for ROI selection
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(data[[2, 1, 0], :, :], (1, 2, 0)))  # RGB from bands
    ax.set_title("Click two corners to define ROI")

    # Connect mouse click events
    cid_click = fig.canvas.mpl_connect('button_press_event', onclick_roi)
    cid_finalize = fig.canvas.mpl_connect('key_press_event', finalize_roi)
    
    plt.show()

    if len(roi_points) < 2:
        print("ROI not defined. Exiting.")
        return

    # Get the rectangular ROI points
    x_coords = [roi_points[0][0], roi_points[1][0]]
    y_coords = [roi_points[0][1], roi_points[1][1]]
    roi = (x_coords, y_coords)

    # Randomly sample points from ROI
    sampled_points = random_sample_points(data, roi)
    print(f"Randomly sampled points: {sampled_points}")

    # Use KNN with selected points
    knn_result = apply_knn(data, sampled_points)

    visualize_results(data, knn_result)

if __name__ == '__main__':
    file_path = r"C:\Users\Pratham Jain\SisterDear\Geospatial\raster\Landsat_ETM_2001-08-26_multispectral.tif"
    main(file_path)
