import rasterio
import numpy as np
import matplotlib.pyplot as plt

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

def compute_ndvi(data):
    """Compute NDVI using Red and NIR bands."""
    red_band = data[2, :, :].astype(float)  # Band 3 (Red)
    nir_band = data[3, :, :].astype(float)  # Band 4 (NIR)
    
    # Avoid division by zero
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)
    return ndvi

def visualize_images(stretched_data, ndvi, data):
    """Visualize the stretched image and NDVI results."""
    image_org = data[[2, 1, 0], :, :].astype(float)  # Convert to float for display
    rgb_stretched = stretched_data[[2, 1, 0], :, :]  # Red, Green, Blue
    rgb_normalized = (rgb_stretched / rgb_stretched.max() * 255).astype(np.uint8)
    rgb_image = np.transpose(rgb_normalized, (1, 2, 0))

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Stretched RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title('Stretched RGB Image')
    axes[0].axis('off')

    # NDVI image
    axes[1].imshow(ndvi, cmap='RdYlGn')  # Colormap for NDVI
    axes[1].set_title('NDVI Image')
    axes[1].axis('off')

    # Original image
    axes[2].imshow(np.transpose(image_org, (1, 2, 0)) / image_org.max())  # Normalize for display
    axes[2].set_title('Original Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def main(file_path):
    """Main function to load data and compute NDVI and stretched image."""
    data = load_tiff_file(file_path)

    # Print the shape of the data for confirmation
    print(f"Data shape: {data.shape}")  # (bands, height, width)

    # Spectral stretching
    stretched_data = spectral_stretch(data)

    # Compute NDVI
    ndvi = compute_ndvi(data)

    # Visualize stretched image and NDVI
    visualize_images(stretched_data, ndvi, data)

if __name__ == '__main__':
    # Path to your .tiff file
    file_path = r"C:\Users\Pratham Jain\SisterDear\Geospatial\raster\Landsat_ETM_2001-08-26_multispectral.tif"
    main(file_path)
