import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from rasterio.plot import show
from collections import Counter

# Function for subnetting the raster (i.e., dividing into smaller regions)
def subnet_raster(data, num_subnets):
    rows, cols = data.shape
    subnet_rows = rows // num_subnets
    subnet_cols = cols // num_subnets
    subnets = []
    for i in range(num_subnets):
        for j in range(num_subnets):
            r_start = i * subnet_rows
            r_end = (i + 1) * subnet_rows
            c_start = j * subnet_cols
            c_end = (j + 1) * subnet_cols
            subnets.append(data[r_start:r_end, c_start:c_end])
    return subnets

# Function for replacing specific values in the raster matrix
def replace_values(data, old_value, new_value):
    data[data == old_value] = new_value
    return data

# Function for segmenting raster data (e.g., thresholding)
def segment_raster(data, threshold):
    return np.where(data > threshold, 1, 0)

# Function to replace the most occurring value in the image
def replace_most_occurring_value(data):
    mean = np.nanmean(data)
    maximum = np.nanargmax(data)
    # Flatten the data array
    flattened_data = data.flatten()
    # Count the frequency of each value
    value_counts = Counter(flattened_data[~np.isnan(flattened_data)])
    # Find the most occurring value
    most_occurring_value = value_counts.most_common(1)[0][0]
    # Replace the most occurring value with  in the original 2D data
    data[data == most_occurring_value] = maximum - mean
    return data, most_occurring_value

# Open the raster file
# dataname = r"C:\Users\Pratham Jain\SisterDear\Geospatial\raster\gm-jpn-el_u_1_1\jpn\el.tif"
dataname = r"C:\Users\Pratham Jain\SisterDear\Geospatial\raster\NextGen_Tokyo_JP_15m_Geo\NextGen_Tokyo_JP_15m_Geo.tif"
with rasterio.open(dataname) as src:
    # Display raster image
    rasterio.plot.show(src, title="Trial Image - Japan Elevation")

    # Print raster metadata
    print(src.bounds, src.count, src.width, src.height, src.crs)

    # Define the bounding box coordinates for Japan province
    left, bottom, right, top = 128.0, 27.02, 149.0, 46.38

    # Get the transform
    transform = src.transform
    
    # Convert bounding box coordinates to pixel coordinates
    col_min, row_max = ~transform * (left, top)
    col_max, row_min = ~transform * (right, bottom)

    # Convert to integer pixel indices
    col_min, row_max = int(col_min), int(row_max)
    col_max, row_min = int(col_max), int(row_min)

    # Swap row_min and row_max to correct window dimensions
    row_min, row_max = row_max, row_min

    # Print pixel coordinates for debugging
    print(f"Pixel Coordinates: col_min={col_min}, row_max={row_max}, col_max={col_max}, row_min={row_min}")

    # Ensure that width and height are non-negative
    if col_max > col_min and row_max > row_min:
        # Define the window
        window = Window(col_off=col_min, row_off=row_min, width=abs(col_max - col_min), height=abs(row_max - row_min))

        # Read the data from the window
        data = src.read(1, window=window)  # Read the first band
        data = data*10

        # Print some information about the data
        print(f"Subsetting window: {window}")
        print(f"Data shape: {data.shape}")

        # Save the visualization as an image
        plt.imshow(data, cmap='Greys')  # Assuming single-band data
        plt.title("Subsetted Raster Data")
        plt.colorbar()
        plt.savefig(r'C:\Users\Pratham Jain\SisterDear\Geospatial\jp_subsetted_data_replaced_visualization.png')
        plt.show()

        # Plot histogram of pixel values
        plt.hist(data[~np.isnan(data)].flatten(), bins=50, color='blue', edgecolor='black')
        plt.title("Histogram of Pixel Values")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.savefig(r'C:\Users\Pratham Jain\SisterDear\Geospatial\jp_pixel_value_histogram.png')
        plt.show()

        # Calculate basic statistics
        mean = np.nanmean(data)
        std_dev = np.nanstd(data)
        print(f"Mean: {mean}, Standard Deviation: {std_dev}")

        # Apply matrix manipulation tasks
        data, most_occurring_value = replace_most_occurring_value(data)
        print(f"Most occurring value: {most_occurring_value}")

        # Define the transform for the new subset
        new_transform = src.transform * Affine.translation(col_min, row_min)

        # Create a new file for the subsetted data
        new_filename = r"C:\Users\Pratham Jain\SisterDear\Geospatial\vector\gm-jpn-pop_u_2\new_builtupa_jpn.tif"
        with rasterio.open(
            new_filename, 'w',
            driver='GTiff',
            height=window.height,
            width=window.width,
            count=1,
            dtype=data.dtype,
            crs=src.crs,
            transform=new_transform
        ) as dst:
            dst.write(data, 1)
    else:
        print("Invalid window dimensions: width and height must be non-negative.")

    # Save the visualization as an image
    plt.imshow(data, cmap='Greys')  # Assuming single-band data
    plt.title("Subsetted Raster Data with replaced values")
    plt.colorbar()
    plt.savefig(r'C:\Users\Pratham Jain\SisterDear\Geospatial\jp_subsetted_data_replaced_visualization.png')
    plt.show()


# Load vector data
vector_file = r"C:\Users\Pratham Jain\SisterDear\Geospatial\vector\gm-jpn-pop_u_2\builtupp_jpn.shp"
vector_data = gpd.read_file(vector_file)

# # Print CRS information for debugging
# print("Vector CRS:", vector_data.crs)
# print("Raster CRS:", src.crs)

# # Convert vector CRS to match raster CRS
# if vector_data.crs != src.crs:
#     try:
#         vector_data = vector_data.to_crs(src.crs)
#         print("Vector CRS:", vector_data.crs , "Raster CRS:" ,src.crs)
#     except Exception as e:
#         print("CRS transformation error:", e)

# # Print basic information about vector data
# print(vector_data.info())

# # Plot vector data
# vector_data.plot()
# plt.title("Vector Data")
# plt.show()

# # Plot raster and vector data together
# fig, ax = plt.subplots(figsize=(10, 10))
# show(data, ax=ax, cmap='inferno', title="Raster with Vector Data")
# vector_data.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)
# plt.show()
# plt.savefig(r'C:\Users\Pratham Jain\SisterDear\Geospatial\jp_Raster_with_Vector_Data.png')
