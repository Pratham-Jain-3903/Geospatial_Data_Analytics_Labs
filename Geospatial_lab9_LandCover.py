import rasterio
import numpy as np
import matplotlib.pyplot as plt

"""
The ETM+ contains eight spectral bands, including a pan and thermal band:
Band 1 Blue (0.45 - 0.52 µm) 30 m
Band 2 Green (0.52 - 0.60 µm) 30 m
Band 3 Red (0.63 - 0.69 µm) 30 m
Band 4 Near-Infrared (0.77 - 0.90 µm) 30 m
Band 5 Short-wave Infrared (1.55 - 1.75 µm) 30 m
Band 6 Thermal (10.40 - 12.50 µm) 60 m Low Gain / High Gain
Band 7 Mid-Infrared (2.08 - 2.35 µm) 30 m
Band 8 Panchromatic (PAN) (0.52 - 0.90 µm) 15 m

"""

def load_tiff_file(file_path):
    """Load the .tiff file and return its content."""
    with rasterio.open(file_path) as src:
        data = src.read()  # Read all bands into a 3D array (bands x height x width)
        return data

def cosine_similarity(a, b):
    """Compute the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def classify_pixels(data, water_avg, land_avg, vegetation_avg):
    """Classify each pixel based on cosine similarity to average values."""
    bands, height, width = data.shape
    classifications = np.full((height, width), -1, dtype=int)  # -1: Unknown, 0: Water, 1: Land, 2: Vegetation

    for i in range(height):
        for j in range(width):
            pixel = data[:, i, j]  # Get pixel across all bands
            sim_water = cosine_similarity(pixel, water_avg)
            sim_land = cosine_similarity(pixel, land_avg)
            sim_vegetation = cosine_similarity(pixel, vegetation_avg)

            # Get the maximum similarity and its corresponding class
            similarities = np.array([sim_water, sim_land, sim_vegetation])
            max_similarity = np.max(similarities)
            threshold = 0.5  # 50% threshold

            # Classify based on highest similarity and threshold
            if max_similarity > threshold:
                classifications[i, j] = np.argmax(similarities)

    return classifications

def visualize_classification(classifications, original_data):
    """Visualize the classification results with original RGB bands."""
    height, width = classifications.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)  # Create an empty RGB image

    # Assign colors based on classifications
    color_image[classifications == 0] = [0, 0, 255]    # Water: Blue
    color_image[classifications == 1] = [255, 0, 0]    # Land: Red
    color_image[classifications == 2] = [0, 255, 0]    # Vegetation: Green
    color_image[classifications == -1] = [0, 0, 0]     # Undetermined: Black

    # Normalize the original RGB bands for visualization
    rgb_bands = original_data[[2, 1, 0], :, :]  # Assuming bands 3, 2, 1 are RGB
    rgb_normalized = (rgb_bands / rgb_bands.max() * 255).astype(np.uint8)
    rgb_image = np.transpose(rgb_normalized, (1, 2, 0))

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the classified image
    ax1.imshow(color_image)
    ax1.set_title('Classified Image')
    ax1.axis('off')

    # Plot the original RGB image
    ax2.imshow(rgb_image)
    ax2.set_title('Original RGB Image')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def main(file_path):
    """Main function to load data and classify pixels."""
    # Load the .tiff file
    data = load_tiff_file(file_path)

    # Print the shape of the data for confirmation
    print(f"Data shape: {data.shape}")  # (bands, height, width)

    # Hardcoded pixel values for classification
    water_pixels = np.array([[364, 549], [364, 550], [364, 551], [363, 551], [364, 551]])  # (y, x)
    land_pixels = np.array([[434, 215], [434, 216], [434, 217], [433, 216], [432, 216]])  # (y, x)
    vegetation_pixels = np.array([[16, 698], [17, 698], [17, 698], [70, 663], [69, 663]])  # (y, x)

    # Compute average for each class
    water_avg = np.mean([data[:, px[0], px[1]] for px in water_pixels], axis=0)
    land_avg = np.mean([data[:, px[0], px[1]] for px in land_pixels], axis=0)
    vegetation_avg = np.mean([data[:, px[0], px[1]] for px in vegetation_pixels], axis=0)

    # Classify all pixels
    classifications = classify_pixels(data, water_avg, land_avg, vegetation_avg)

    # Visualize the classification result alongside the original RGB image
    visualize_classification(classifications, data)

if __name__ == '__main__':
    # Path to your .tiff file
    file_path = r"C:\Users\Pratham Jain\SisterDear\Geospatial\raster\Landsat_ETM_2001-08-26_multispectral.tif"
    main(file_path)