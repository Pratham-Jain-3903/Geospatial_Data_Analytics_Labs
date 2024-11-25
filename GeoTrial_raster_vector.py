# import rasterio
# from rasterio.windows import Window
# import rasterio.plot
# from rasterio.transform import Affine
# import matplotlib.pyplot as plt
# import numpy as np
# import geopandas as gpd
# from rasterio.plot import show

# # Open the raster file
# dataname = r"C:\Users\Pratham Jain\SisterDear\Geospatial\raster\gm-jpn-el_u_1_1\jpn\el.tif"
# with rasterio.open(dataname) as src:
#     # Display raster image
#     rasterio.plot.show(src, title="Trial Image - Tokyo")

#     # Print raster metadata
#     print(src.bounds, src.count, src.width, src.height, src.crs)

#     # Define the bounding box coordinates for Tokyo province
#     # left, bottom, right, top = 139.39, 35.159946, 140.3901928, 36.16

#     left, bottom, right, top = 128.0, 27.02, 149.0, 45.4

#     # Get the transform
#     transform = src.transform
    
#     # Convert bounding box coordinates to pixel coordinates
#     col_min, row_max = ~transform * (left, top)
#     col_max, row_min = ~transform * (right, bottom)

#     # Convert to integer pixel indices
#     col_min, row_max = int(col_min), int(row_max)
#     col_max, row_min = int(col_max), int(row_min)

#     # Swap row_min and row_max to correct window dimensions
#     row_min, row_max = row_max, row_min

#     # Print pixel coordinates for debugging
#     print(f"Pixel Coordinates: col_min={col_min}, row_max={row_max}, col_max={col_max}, row_min={row_min}")

#     # Ensure that width and height are non-negative
#     if col_max > col_min and row_max > row_min:
#         # Define the window
#         window = Window(col_off=col_min, row_off=row_min, width=abs(col_max - col_min), height=abs(row_max - row_min))

#         # Read the data from the window
#         data = src.read(window=window)

#         # Print some information about the data
#         print(f"Subsetting window: {window}")
#         print(f"Data shape: {data.shape}")

#         # Save the visualization as an image
#         plt.imshow(data[0], cmap='gray')  # Assuming single-band data; adjust if multi-band
#         plt.title("Subsetted Raster Data")
#         plt.colorbar()
#         plt.savefig(r'C:\Users\Pratham Jain\SisterDear\Geospatial\jp_subsetted_data_visualization.png')
#         plt.show()

#         # Calculate basic statistics
#         mean = np.mean(data)
#         std_dev = np.std(data)
#         print(f"Mean: {mean}, Standard Deviation: {std_dev}")

#         # Define the transform for the new subset
#         new_transform = src.transform * Affine.translation(col_min, row_min)

#         # Create a new file for the subsetted data
#         new_filename = r"C:\Users\Pratham Jain\SisterDear\Geospatial\vector\gm-jpn-pop_u_2\new_builtupa_jpn.tif"
#         with rasterio.open(
#             new_filename, 'w',
#             driver='GTiff',
#             height=window.height,
#             width=window.width,
#             count=src.count,
#             dtype=data.dtype,
#             crs=src.crs,
#             transform=new_transform
#         ) as dst:
#             dst.write(data)
#     else:
#         print("Invalid window dimensions: width and height must be non-negative.")

# # Load vector data
# vector_file = r"C:\Users\Pratham Jain\SisterDear\Geospatial\vector\gm-jpn-pop_u_2\builtupp_jpn.shp"
# vector_data = gpd.read_file(vector_file)

# # Print CRS information for debugging
# print("Vector CRS:", vector_data.crs)
# print("Raster CRS:", src.crs)

# # Convert vector CRS to match raster CRS
# if vector_data.crs != src.crs:
#     try:
#         vector_data = vector_data.to_crs(src.crs)
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
# show(data[0], ax=ax, cmap='gray', title="Raster with Vector Data")
# vector_data.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)
# plt.show()
# plt.savefig(r'C:\Users\Pratham Jain\SisterDear\Geospatial\jp_Raster_with_Vector_Data.png')

import rasterio
from rasterio.windows import Window
import rasterio.plot
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from rasterio.plot import show

# Open the raster file
dataname = r"C:\Users\Pratham Jain\SisterDear\Geospatial\raster\gm-jpn-el_u_1_1\jpn\el.tif"
with rasterio.open(dataname) as src:
    # Display raster image
    rasterio.plot.show(src, title="Trial Image - Japan Elevation")

    # Print raster metadata
    print(src.bounds, src.count, src.width, src.height, src.crs)

    # Define the bounding box coordinates for Tokyo province
    left, bottom, right, top = 128.0, 27.02, 149.0, 45.4

    # Get the transform
    transform = src.transform
    print("transform",transform)
    
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
        data = src.read(window=window)

        # Print some information about the data
        print(f"Subsetting window: {window}")
        print(f"Data shape: {data.shape}")

        # Save the visualization as an image
        plt.imshow(data[0], cmap='inferno')  # Assuming single-band data; adjust if multi-band 
        plt.title("Subsetted Raster Data")
        plt.colorbar()
        plt.savefig(r'C:\Users\Pratham Jain\SisterDear\Geospatial\jp_subsetted_data_visualization.png')
        plt.show()

        # Calculate basic statistics
        mean = np.mean(data)
        std_dev = np.std(data)
        print(f"Mean: {mean}, Standard Deviation: {std_dev}")

        # Define the transform for the new subset
        new_transform = src.transform * Affine.translation(col_min, row_min)

        # Create a new file for the subsetted data
        new_filename = r"C:\Users\Pratham Jain\SisterDear\Geospatial\vector\gm-jpn-pop_u_2\new_builtupa_jpn.tif"
        with rasterio.open(
            new_filename, 'w',
            driver='GTiff',
            height=window.height,
            width=window.width,
            count=src.count,
            dtype=data.dtype,
            crs=src.crs,
            transform=new_transform
        ) as dst:
            dst.write(data)
    else:
        print("Invalid window dimensions: width and height must be non-negative.")

# Load vector data
vector_file = r"C:\Users\Pratham Jain\SisterDear\Geospatial\vector\gm-jpn-pop_u_2\builtupp_jpn.shp"
vector_data = gpd.read_file(vector_file)

# Print CRS information for debugging
print("Vector CRS:", vector_data.crs)
print("Raster CRS:", src.crs)

# Convert vector CRS to match raster CRS
if vector_data.crs != src.crs:
    try:
        vector_data = vector_data.to_crs(src.crs)
        print("Vector CRS:", vector_data.crs , "Raster CRS:" ,src.crs)
    except Exception as e:
        print("CRS transformation error:", e)

# Print basic information about vector data
print(vector_data.info())

# Plot vector data
# vector_data = vector_data*(100)
vector_data.plot()
plt.title("Vector Data")
plt.show()

# Plot raster and vector data together
fig, ax = plt.subplots(figsize=(10, 10))
show(data[0], ax=ax, cmap='inferno', title="Raster with Vector Data")
vector_data.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)
plt.show()
plt.savefig(r'C:\Users\Pratham Jain\SisterDear\Geospatial\jp_Raster_with_Vector_Data.png')
