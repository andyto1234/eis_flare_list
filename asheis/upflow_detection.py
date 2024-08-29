import pandas as pd
import requests
from pathlib import Path
import matplotlib.pyplot as plt
from sunpy.map import Map
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from tqdm import tqdm
import glob
import sys
from IPython.display import clear_output
from . import asheis

# Read the CSV file
csv_path = 'database/downloaded_flare_data.csv'
df = pd.read_csv(csv_path)

import numpy as np
from scipy import ndimage
from skimage import filters, morphology

def find_largest_upflow_region(dopp_eis):
    # Extract data from the EIS map
    data = dopp_eis.data
    
    # Pre-processing: Remove positive velocities and extreme negative velocities
    data = np.where(data > 0, 0, data)
    data = np.where(data < -55, 0, data)
    
    # Smoothing: Apply Gaussian filter to reduce noise
    smoothed_data = ndimage.gaussian_filter(data, sigma=1.5, mode='nearest')
    
    # Further noise reduction: Apply median filter
    median_filtered = filters.median(smoothed_data, morphology.disk(3))
    
    # Create binary mask for upflow regions
    upflow_mask = (median_filtered < -10)
    
    # Clean up the mask: Remove small objects and close gaps
    cleaned = morphology.remove_small_objects(upflow_mask, min_size=40)
    cleaned = morphology.closing(cleaned, morphology.disk(3))
    
    # Label connected regions in the cleaned mask
    labeled_mask, num_features = ndimage.label(cleaned)
    
    if num_features > 0:
        # Find the largest connected region
        sizes = ndimage.sum(cleaned, labeled_mask, range(1, num_features + 1))
        largest_feature_label = sizes.argmax() + 1
        largest_feature_mask = labeled_mask == largest_feature_label
        
        # Get the bounding box of the largest region
        y, x = np.where(largest_feature_mask)
        bottom_left_y, bottom_left_x = np.min(y), np.min(x)
        top_right_y, top_right_x = np.max(y), np.max(x)
        
        # Calculate spans
        x_span = top_right_x - bottom_left_x
        y_span = top_right_y - bottom_left_y
        
        # Define shrink factor function
        def get_shrink_factor(span):
            if span <= 8:
                return 1.0  # No shrinking for small spans
            elif span <= 15:
                return 0.9  # 10% shrink for medium spans
            else:
                return max(0.5, 0.8 - (span - 15) * 0.01)  # More aggressive shrinking for larger spans, minimum 50%
        
        # Calculate shrink factors
        shrink_factor_x = get_shrink_factor(x_span)
        shrink_factor_y = get_shrink_factor(y_span)
        
        # Calculate center points
        center_x = (bottom_left_x + top_right_x) / 2
        center_y = (bottom_left_y + top_right_y) / 2
        
        # Apply shrinking
        new_x_span = x_span * shrink_factor_x
        new_y_span = y_span * shrink_factor_y
        
        # Calculate new bounding box
        bottom_left_x = int(center_x - new_x_span / 2)
        bottom_left_y = int(center_y - new_y_span / 2)
        top_right_x = int(center_x + new_x_span / 2)
        top_right_y = int(center_y + new_y_span / 2)
        
        # Convert pixel coordinates to world coordinates
        bottom_left_world = dopp_eis.pixel_to_world((bottom_left_x) * u.pix, (bottom_left_y) * u.pix)
        top_right_world = dopp_eis.pixel_to_world((top_right_x) * u.pix, (top_right_y) * u.pix)
        
        return bottom_left_world, top_right_world
    else:
        print("No significant upflow region found.")
        return None, None

# Example usage:
# bottom_left_world, top_right_world = find_largest_upflow_region(dopp_eis)
# if bottom_left_world and top_right_world:
#     print(f"World coordinates of the bottom left: {bottom_left_world}")
#     print(f"World coordinates of the top right: {top_right_world}")
# else:
#     print("No significant upflow region found.")
# Function to download file with progress bar
def download_file(url, save_path):
    max_retries = 10
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            with open(save_path, 'wb') as f, tqdm(
                desc=save_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    progress_bar.update(size)
            
            print(f"Downloaded: {save_path}")
            return  # Success, exit the function
        
        except (requests.exceptions.RequestException, OSError) as e:
            if attempt < max_retries - 1:
                print(f"Error occurred: {str(e)}. Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
            else:
                print(f"Failed to download after {max_retries} attempts: {url}")
                print(f"Error: {str(e)}")
                raise  # Re-raise the exception to be handled by the caller

# Process each entry in the CSV
for index, row in df.iterrows():    
    # Clear output
    clear_output(wait=True)
    
    # Construct file names using the filename entry
    data_file = row['filename']
    year = data_file[4:8]
    month = data_file[8:10]
    date = data_file[10:12]
    time = data_file.split('_')[2][:6]

    # Check if the plot already exists
    download_dir = Path('data_eis')
    plot_path = download_dir / f"doppler_velocity_map_{year}_{month}_{date}_{time}.png"
    no_upflow_plot_path = download_dir / f"no_upflow_doppler_velocity_map_{year}_{month}_{date}_{time}.png"
    
    if plot_path.exists() or no_upflow_plot_path.exists():
        print(f"Plot already exists for {year}-{month}-{date} {time}. Skipping...")
        continue

    head_file = data_file.replace('.data.h5', '.head.h5')
    
    # Construct URLs
    base_url = f"https://vsolar.mssl.ucl.ac.uk/eispac/hdf5/{year}/{month}/{date}/"
    data_url = base_url + data_file
    head_url = base_url + head_file
    
    # Create directory for downloads if it doesn't exist
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Download files
    data_path = download_dir / data_file
    head_path = download_dir / head_file
    print(f"Downloading files for {year}-{month}-{date} {time}")
    download_file(data_url, data_path)
    download_file(head_url, head_path)
    
    try:
        # Process the downloaded files
        print(f"Processing files for {year}-{month}-{date} {time}")
        eis_map = asheis.asheis(data_path)
        
        # Get Doppler velocity
        dopp_eis = eis_map.get_velocity('fe_12_195.12', plot=False)
        
        # Find largest upflow region
        bottom_left_world, top_right_world = find_largest_upflow_region(dopp_eis)
        
        # Create a new figure with WCS projection
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection=dopp_eis)
        
        # Plot the Doppler velocity map
        im = dopp_eis.plot(axes=ax, norm=plt.Normalize(vmin=-15, vmax=15), cmap='RdBu_r')
        
        if bottom_left_world and top_right_world:
            # Plot the largest upflow region
            coords = SkyCoord(Tx=(bottom_left_world.Tx.value, top_right_world.Tx.value) * u.arcsec, 
                              Ty=(bottom_left_world.Ty.value, top_right_world.Ty.value) * u.arcsec, 
                              frame=dopp_eis.coordinate_frame)
            
            dopp_eis.draw_quadrangle(coords, axes=ax, edgecolor="red", lw=2)
            
            plot_prefix = ""
        else:
            print(f"No significant upflow region found for {year} {month} {date} {time}")
            plot_prefix = "no_upflow_"
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Doppler Velocity [km/s]')
        
        # Set title
        plt.title(f"Doppler Velocity Map - {year} {month} {date} {time}")
        
        # Save the plot
        plot_path = download_dir / f"{plot_prefix}doppler_velocity_map_{year}_{month}_{date}_{time}.png"
        plt.savefig(plot_path)
        plt.close(fig)
        
        print(f"Plot saved: {plot_path}")
    
    except Exception as e:
        print(f"Error processing {year} {month} {date} {time}: {str(e)}")
    
    finally:
        # Delete the downloaded files and additional .fit.h5 files
        files_to_delete = [
            data_path,
            head_path,
            *glob.glob(str(download_dir / f"eis_{year}{month}{date}_{time}*.fit.h5"))
        ]
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
