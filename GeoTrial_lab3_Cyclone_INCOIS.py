import tifffile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D

# Define the list of image files to process
image_files = [
    r"C:\Users\Pratham Jain\SisterDear\Geospatial\Cyclone_titli\20181007-163200Z-metop-1-cloudtopbt4.tif", 
    r"C:\Users\Pratham Jain\SisterDear\Geospatial\Cyclone_titli\20181008-161110Z-metop-1-cloudtopbt4.tif",
    r"C:\Users\Pratham Jain\SisterDear\Geospatial\Cyclone_titli\20181009-173420Z-metop-1-cloudtopbt4.tif",
    r"C:\Users\Pratham Jain\SisterDear\Geospatial\Cyclone_titli\20181010-171130Z-metop-1-cloudtopbt4.tif",
    r"C:\Users\Pratham Jain\SisterDear\Geospatial\Cyclone_titli\20181011-164950Z-metop-1-cloudtopbt4.tif",
    r"C:\Users\Pratham Jain\SisterDear\Geospatial\Cyclone_titli\20181012-162830Z-metop-1-cloudtopbt4.tif"
]

# Define a minimum temperature threshold
min_temperature_threshold = -50  

# Create a list to store the processed temperature data for each day
temperature_series = []

# Loop through each image file and process the temperature data
for file in image_files:
    # Open the TIFF file
    with tifffile.TiffFile(file) as tif:
        # Read the image data
        image_data = tif.asarray()

    # Set minimum temperature values to NaN
    temperature_values = np.where(image_data <= min_temperature_threshold, np.nan, image_data)

    # Append the processed temperature data to the list
    temperature_series.append(temperature_values)

# Save the images produced from heatmaps
image_filenames = []
for i, temp_data in enumerate(temperature_series):
    plt.figure(figsize=(10, 8))
    sns.heatmap(temp_data, cmap='coolwarm', cbar=True)
    plt.title(f"Temperature on {image_files[i][-22:-4]}")  # Extracting date from filename
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    image_filename = f"temperature_frame_{i}.png"
    plt.savefig(image_filename)
    plt.close()
    image_filenames.append(image_filename)

# Set up the figure for the animation
fig, ax = plt.subplots(figsize=(10, 8))

def update_plot(frame):
    # Clear the previous frame
    ax.clear()
    # Load and display the current frame image
    img = plt.imread(image_filenames[frame])
    ax.imshow(img)
    ax.set_title(f"Temperature on {image_files[frame][-22:-4]}")  # Extracting date from filename
    ax.axis('off')  # Turn off axis

# Create the animation
ani = animation.FuncAnimation(fig, update_plot, frames=len(image_files), interval=1000, repeat=False)

# Save the animation as a GIF
ani.save('temperature_animation.gif', writer=PillowWriter(fps=1))

# Show the animation
plt.show()
