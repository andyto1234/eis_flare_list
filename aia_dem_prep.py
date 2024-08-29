import os
import glob
import numpy as np
import sunpy.map
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import time_support
from aiapy.calibrate import estimate_error
from useful_packages.useful_sdo import get_prepped_aia_maps

# This lets you pass `astropy.time.Time` objects directly to matplotlib
time_support(format="jyear")

def create_submaps(aia_maps, width_map):
    # width_buffer = np.abs(width_map.top_right_coord.Tx - width_map.bottom_left_coord.Tx)*0.1
    # height_buffer = np.abs(width_map.top_right_coord.Ty - width_map.bottom_left_coord.Ty)*0.1

    # bottom_left = SkyCoord(
    #     int(width_map.bottom_left_coord.Tx.value - width_buffer.value)*u.arcsec,
    #     int(width_map.bottom_left_coord.Ty.value - height_buffer.value)*u.arcsec,
    #     frame=aia_maps[94].coordinate_frame
    # )

    # top_right = SkyCoord(
    #     int(width_map.top_right_coord.Tx.value + width_buffer.value)*u.arcsec,
    #     int(width_map.top_right_coord.Ty.value + height_buffer.value)*u.arcsec,
    #     frame=aia_maps[94].coordinate_frame
    # )
    print(f"Submap coordinates calculated with 10% buffer.")

    for wavelength in aia_maps.keys():
        # print(bottom_left, top_right)
        # print(aia_maps[wavelength])
        # aia_maps[wavelength] = aia_maps[wavelength].submap(bottom_left, top_right=top_right)
        aia_maps[wavelength] = aia_maps[wavelength].submap(width_map.bottom_left_coord, top_right=width_map.top_right_coord)

    # width_map = width_map.submap(bottom_left, top_right=top_right)

    print("All maps have been submap'ed to the same region.")
    return aia_maps

def save_aia_maps(aia_maps, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for wavelength, map in aia_maps.items():
        file_name = f'aia_map_{map.date.strftime("%Y%m%d_%H%M%S")}_{wavelength}A.fits'
        file_path = os.path.join(save_dir, file_name)
        map.save(file_path, overwrite=True)
        print(f"Map for {wavelength}A saved to {file_path}")

    print("All AIA maps have been saved.")

def read_saved_aia_maps(save_dir):
    saved_maps = glob.glob(os.path.join(save_dir, 'aia_map_*A.fits'))
    saved_aia_maps = {}

    for saved_map in saved_maps:
        wavelength = saved_map.split('_')[-1][:-1].split('A')[0]
        map = sunpy.map.Map(saved_map)
        saved_aia_maps[wavelength] = map

    print("All saved AIA maps have been read.")
    return saved_aia_maps

def prepare_aia_data(saved_aia_maps):
    wavelength_list = ['94','131','171','193','211','335']

    image_array = np.zeros((saved_aia_maps['94'].data.shape[0], saved_aia_maps['94'].data.shape[1], len(wavelength_list)))
    error_array = image_array.copy()

    for i, wavelength in enumerate(wavelength_list):
        num_pix = saved_aia_maps[wavelength].data.size
        error_array[:,:,i] = estimate_error(saved_aia_maps[wavelength].data*(u.ct/u.pix), saved_aia_maps[wavelength].wavelength)
        image_array[:,:,i] = saved_aia_maps[wavelength].data

    return image_array, error_array

def rebin_array(array, factor):
    """
    Rebin a 3D array by a given factor.

    Parameters:
    array (numpy.ndarray): The input 3D array to be rebinned.
    factor (int): The rebinning factor.

    Returns:
    numpy.ndarray: The rebinned 3D array.
    """
    shape = (array.shape[0] // factor, factor,
             array.shape[1] // factor, factor,
             array.shape[2])
    return array.reshape(shape).mean(axis=(1, 3))

def get_aia_dem_prep(reference_map, save_dir, rebin_factor=False,map_date=None):
    # Get prepared AIA maps
    if map_date:
        aia_maps = get_prepped_aia_maps(map_date)
    else:
        aia_maps = get_prepped_aia_maps(reference_map.date)

    # Create submaps
    aia_maps = create_submaps(aia_maps, reference_map)

    # Save AIA maps
    save_aia_maps(aia_maps, save_dir)

    # Read saved AIA maps
    saved_aia_maps = read_saved_aia_maps(save_dir)

    # Sort AIA data for DEM calculation
    image_array, error_array = prepare_aia_data(saved_aia_maps)

    if rebin_factor:
        image_array = rebin_array(image_array, rebin_factor)
        error_array = rebin_array(error_array, rebin_factor)

    return image_array, error_array, saved_aia_maps

if __name__ == "__main__":
    # You would typically get these from command line arguments or a configuration file
    width_map = None  # You need to define this
    save_dir = '/Users/andysh.to/Script/Data/IRIS_output/20190413_113910/20190413_113910/saved_maps'

    image_array, error_array = get_aia_dem_prep(width_map, save_dir)
    print("AIA data preparation complete.")
    print(f"Image array shape: {image_array.shape}")
    print(f"Error array shape: {error_array.shape}")