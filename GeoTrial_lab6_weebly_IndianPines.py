import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math

def load_mat_file(file_path):
    """Load the .mat file and return its content."""
    try:
        # Try loading with scipy for older MATLAB formats
        mat = scipy.io.loadmat(file_path)
    except NotImplementedError:
        # Fallback to h5py for MATLAB v7.3 or later
        mat = h5py.File(file_path, 'r')
    return mat

def print_band_names(mat):
    """Print all variable names in the .mat file."""
    if isinstance(mat, dict):
        # Scipy's dict-based format
        for key in mat:
            print(f'Variable name: {key}')
    elif isinstance(mat, h5py.File):
        # H5py's key-based format
        for key in mat.keys():
            print(f'Variable name: {key}')

def visualize_all_bands(data):
    """Display all bands of the hyperspectral data in a single plot."""
    num_bands = data.shape[2]  # as data is 3D (height x width x bands)
    
    # Determine grid size
    cols = 20  # Number of columns in the grid
    rows = math.ceil(num_bands / cols)  # Number of rows in the grid

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    fig.suptitle('Hyperspectral Data Bands', fontsize=10)

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Plot each band
    for band in range(num_bands):
        ax = axes[band]
        ax.imshow(data[:, :, band], cmap='Greens')
        ax.set_title(f'Band {band}')
        ax.axis('off')

    # Hide any unused subplots
    for i in range(num_bands, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_histograms(data, bands):
    """Plot histograms for selected bands."""
    for band in bands:
        plt.figure()
        plt.hist(data[:, :, band].ravel(), bins=50, color='blue', edgecolor='black')
        plt.title(f'Histogram of Band {band}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()

def main(file_path, bands_to_plot):
    """Main function to load data, print keys, visualize, and plot histograms."""
    # Load the .mat file
    mat = load_mat_file(file_path)

    # Print all keys to find the correct one for hyperspectral data
    print("Available keys in the .mat file:")
    print_band_names(mat)

    # Extract hyperspectral data array using the user-specified key
    data_key = 'indian_pines_corrected'

    if isinstance(mat, dict):
        if data_key in mat:
            data = mat[data_key]
        else:
            print(f"Key '{data_key}' not found in the .mat file.")
            return
    elif isinstance(mat, h5py.File):
        if data_key in mat:
            data = mat[data_key][()]
        else:
            print(f"Key '{data_key}' not found in the .mat file.")
            return

    # Print the shape of the data for confirmation
    print(f"Data shape: {data.shape}")

    # Visualize all bands
    visualize_all_bands(data)

    # Plot histograms for selected bands
    plot_histograms(data, bands_to_plot)

if __name__ == '__main__':
    # Path to your .mat file
    file_path = r"C:\Users\Pratham Jain\SisterDear\Geospatial\Indian_pines_corrected.mat"
    # List of bands to plot histograms for
    bands_to_plot = [0, 1, 100, 199]  # Example: First three bands
    main(file_path, bands_to_plot)
