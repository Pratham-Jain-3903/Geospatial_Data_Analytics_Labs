import rasterio
import numpy as np
import matplotlib.pyplot as plt
import math

def load_tiff_file(file_path):
    """Load the .tiff file and return its content."""
    with rasterio.open(file_path) as src:
        data = src.read()  # Read all bands into a 3D array (bands x height x width)
        return data

def visualize_all_bands(data):
    """Display all bands of the hyperspectral data in a single plot."""
    num_bands = data.shape[0]  # data is 3D (bands x height x width)
    
    # Determine grid size
    cols = 5  # Number of columns in the grid
    rows = math.ceil(num_bands / cols)  # Number of rows in the grid

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    fig.suptitle('Hyperspectral Data Bands', fontsize=10)

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Plot each band
    for band in range(num_bands):
        ax = axes[band]
        ax.imshow(data[band], cmap='terrain_r')
        ax.set_title(f'Band {band + 1}')
        ax.axis('off')

    # Hide any unused subplots
    for i in range(num_bands, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def create_true_color_composite(data):
    """Create and return true color composite (using RGB bands)."""
    # Extract RGB bands based on the specified indices for Blue, Green, Red
    rgb_indices = (0, 1, 2)  # Band 1: Blue, Band 2: Green, Band 3: Red
    rgb = np.dstack([data[i] for i in rgb_indices])

    # Normalize the RGB composite for display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    return rgb

def plot_true_color_composite(data):
    """Plot true color composite in a single plot."""
    true_color = create_true_color_composite(data)

    plt.figure(figsize=(10, 10))
    plt.imshow(true_color)
    plt.title('True Color Composite')
    plt.axis('off')
    plt.show()

def plot_histograms(data, bands):
    """Plot histograms for selected bands in a single plot."""
    plt.figure(figsize=(12, 8))
    
    for band in bands:
        plt.hist(data[band].ravel(), bins=50, alpha=0.5, label=f'Band {band + 1}')

    plt.title('Histograms of Selected Bands')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.show()

def calculate_spectral_signatures(data, class_mask):
    """Calculate spectral signatures for each class."""
    spectral_signatures = {}
    
    for class_name, mask in class_mask.items():
        # Compute mean spectral values for the masked class
        mean_spectrum = np.mean(data[:, mask], axis=1)
        spectral_signatures[class_name] = mean_spectrum
    
    return spectral_signatures

def plot_spectral_signatures(spectral_signatures):
    """Plot spectral signatures for each class."""
    plt.figure(figsize=(10, 6))

    for class_name, spectrum in spectral_signatures.items():
        plt.plot(spectrum, label=class_name)

    plt.title('Spectral Signatures')
    plt.xlabel('Band Index')
    plt.ylabel('Mean Reflectance')
    plt.legend()
    plt.grid()
    plt.show()

def main(file_path, bands_to_plot):
    """Main function to load data, visualize bands, and plot histograms."""
    # Load the .tiff file
    data = load_tiff_file(file_path)

    # Print the shape of the data for confirmation
    print(f"Data shape: {data.shape}")  # (bands, height, width)

    # Visualize all bands
    visualize_all_bands(data)

    # Plot the true color composite
    plot_true_color_composite(data)

    # Plot histograms for selected bands
    plot_histograms(data, bands_to_plot)

    # Example masks for classes: Modify based on your criteria for water, soil, vegetation
    height, width = data.shape[1], data.shape[2]
    water_mask = data[0] > 0.2  # Placeholder condition for water
    soil_mask = data[1] > 0.2   # Placeholder condition for soil
    vegetation_mask = data[2] > 0.2  # Placeholder condition for vegetation

    class_masks = {
        'Water': water_mask.flatten(),
        'Soil': soil_mask.flatten(),
        'Vegetation': vegetation_mask.flatten()
    }

    # Calculate spectral signatures
    spectral_signatures = calculate_spectral_signatures(data, class_masks)

    # Plot spectral signatures
    plot_spectral_signatures(spectral_signatures)

if __name__ == '__main__':
    # Path to your .tiff file
    file_path = r"C:\Users\Pratham Jain\SisterDear\Geospatial\raster\Landsat_ETM_2001-08-26_multispectral.tif"
    # List of bands to plot histograms for
    bands_to_plot = [0, 1, 2, 3, 4, 5, 6]  # Adjust indices based on your data
    main(file_path, bands_to_plot)
