import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.map import Map
from tqdm import tqdm
import os
from pathlib import Path
from get_upflow_region import download_file
from asheis.asheis import asheis
from useful_packages.align_aia_EIS import alignment
import glob

# Read the CSV file
csv_path = 'abbys_flare_data/downloaded_flare_data.csv'
df = pd.read_csv(csv_path)

# Create a directory to store downloaded files
download_dir = Path('downloaded_fits')
download_dir.mkdir(parents=True, exist_ok=True)

# Initialize new columns
df['hpc_x_pixel'] = None
df['hpc_y_pixel'] = None
df['Txshift'] = None
df['Tyshift'] = None

if __name__ == "__main__":
    # Process each row
    for index, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        hpc_x = row['hpc x']
        hpc_y = row['hpc y']
        
        # Construct the URLs
        year = filename[4:8]
        month = filename[8:10]
        date = filename[10:12]
        base_url = f"https://vsolar.mssl.ucl.ac.uk/eispac/hdf5/{year}/{month}/{date}/"
        data_url = base_url + filename
        head_url = base_url + filename.replace('.data.h5', '.head.h5')
        
        # Download the files using the existing download function
        data_save_path = download_dir / filename
        head_save_path = download_dir / filename.replace('.data.h5', '.head.h5')
        download_file(data_url, data_save_path)
        download_file(head_url, head_save_path)
        
        # Check if both files were downloaded successfully
        if data_save_path.exists() and head_save_path.exists():
            try:
                # Create a SunPy Map object
                eis_map = asheis(data_save_path).get_intensity('fe_12_195.12')
                m_eis_fixed, Txshift, Tyshift = alignment(eis_map, return_shift=True)
                # Create a SkyCoord object
                coord = SkyCoord(hpc_x * u.arcsec, hpc_y * u.arcsec, frame=m_eis_fixed.coordinate_frame)
                
                # Convert world coordinates to pixel coordinates
                pixel_coord = m_eis_fixed.world_to_pixel(coord)
                
                # Update the DataFrame
                df.at[index, 'hpc_x_pixel'] = pixel_coord.x.value
                df.at[index, 'hpc_y_pixel'] = pixel_coord.y.value
                df.at[index, 'Txshift'] = Txshift.value
                df.at[index, 'Tyshift'] = Tyshift.value
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
            
            finally:
                # Delete the downloaded files and additional .fit.h5 files
                files_to_delete = [
                    data_save_path,
                    head_save_path,
                    *glob.glob(str(download_dir / f"eis_{year}{month}{date}_*.fit.h5"))
                ]
                
                for file_path in files_to_delete:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
        else:
            print(f"Failed to download {filename} or its head file")
        
        # Save the updated DataFrame after each file is processed
        df.to_csv(csv_path, index=False)
        print(f"CSV file updated after processing {filename}")

    print("Processing complete. CSV file updated with pixel coordinates.")
