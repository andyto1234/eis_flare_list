import os
import json
import pandas as pd
import eispac
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.io import fits
import sunpy.coordinates
import sunpy.map
import sunpy.data.sample
from sunpy.net import Fido
from sunpy.net import attrs as a
from asheis import asheis
from sunkit_image.coalignment import _calculate_shift as calculate_shift
from datetime import datetime, timedelta
from sunpy import timeseries as ts
from useful_packages.align_aia_EIS import alignment
from tqdm import tqdm
import multiprocessing
from scipy.ndimage import gaussian_filter
from astropy.coordinates import SkyCoord
from numba import jit


# Helper Functions
def save_dataframe(df, YEAR):
    print(YEAR)
    filename=f"filtered_results_{YEAR}.csv"
    print(filename)
    df.to_csv(filename, index=False)
    print(f"DataFrame saved to {filename}")

def find_flare_location(hek_entry):
    downloaded_files = None
    try:
        time_delta = TimeDelta(30, format='sec')
        start_time = Time(hek_entry['Start Time'])
        peak_time = Time(hek_entry['Peak Time'])
        start_time_lagged = start_time + time_delta
        peak_time_lagged = peak_time + time_delta
        
        result_start = Fido.search(a.Time(start_time, start_time_lagged),
                                   a.Instrument("aia"), a.Wavelength(94*u.angstrom))[0][0]
        result_peak = Fido.search(a.Time(peak_time, peak_time_lagged),
                                  a.Instrument("aia"), a.Wavelength(94*u.angstrom))[0][0]
        
        downloaded_files = Fido.fetch(result_start, result_peak)
        maps = sunpy.map.Map(downloaded_files)
        
        start_map = maps[0] / maps[0].exposure_time
        end_map = maps[1] / maps[1].exposure_time
        diff = end_map.data - start_map.data
        
        blurred = gaussian_filter(diff, sigma=20)
        diff_map = sunpy.map.Map(blurred, start_map.meta)
        
        pixel_pos = np.argwhere(diff_map.data == diff_map.data.max()) * u.pixel
        hpc_max = diff_map.wcs.pixel_to_world(pixel_pos[:, 1], pixel_pos[:, 0])
        
        return hpc_max.Tx.value[0], hpc_max.Ty.value[0]
    except Exception as e:
        print(f"Error finding flare location for entry {hek_entry}: {e}")
        return None, None
    finally:
        if downloaded_files:
            for file in downloaded_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error removing file {file}: {e}")

def download_file(file):
    if not os.path.exists(f'data_eis/{file}'):
        eispac.download.download_hdf5_data(file)

def parallel_download(filenames, num_processes=5):
    with multiprocessing.Pool(num_processes) as pool:
        list(tqdm(pool.imap(download_file, filenames), total=len(filenames), desc="Downloading files"))

def download_aia_data(start_date_str, end_date_str, wavelength, cadence):
    try:
        res = Fido.search(a.Time(start_date_str, end_date_str), 
                          a.Instrument('aia'), 
                          a.Wavelength(wavelength * u.angstrom), 
                          a.Sample(cadence * u.minute))
        return Fido.fetch(res, path="/disk/solar0/st3/{instrument}/{file}")
    except Exception as e:
        print(f"An error occurred during AIA download: {str(e)}")
        return []

def download_goes_data(start_date_str, end_date_str):
    year = int(start_date_str.split('-')[0])
    sat_no = 15 if year < 2017 else 16 if year < 2023 else 17
    try:
        res = Fido.search(a.Time(start_date_str, end_date_str), 
                          a.Instrument('XRS'),  
                          a.goes.SatelliteNumber(sat_no))
        if len(res[0]) == 0:
            res = Fido.search(a.Time(start_date_str, end_date_str), 
                  a.Instrument('XRS'),  
                  a.goes.SatelliteNumber(sat_no-1))

        return Fido.fetch(res, path="./{instrument}/{file}")
    except Exception as e:
        print(f"An error occurred during GOES download: {str(e)}")
        return []

def process_eis_file(file, hpc_max, goes_class):

    # get filename
    basename = os.path.basename(file)
    date_time_str = basename.split('_')[1] + basename.split('_')[2].split('.')[0]
    date_time_obj = datetime.strptime(date_time_str, '%Y%m%d%H%M%S')
    class_string = goes_class.replace('.','_')
    filename = f"output_{date_time_obj.strftime('%Y%m%d_%H%M%S')}_{class_string}.png"
    output_filename = f"output_pics/{filename}"

    # try:
    # skipping file if it's already been plotted
    if os.path.exists(output_filename):
        # os.remove(file)
        print(f"Skipping... {filename} has been processed")

    if not os.path.exists(output_filename):
        try:
            asheis_file = asheis(file)
            print(f"asheis_file created successfully for {file}")
            
            asheis_map = asheis_file.get_intensity('fe_12_195')
            print(f"asheis_map created successfully. Shape: {asheis_map.data.shape}")
            
            fits_filename = file.replace("data_eis", "fitted_data/fits").replace(".data.h5", "_intensity.fits")
            asheis_map.save(fits_filename, overwrite=True)
            print(f"asheis_map saved to {fits_filename}")
            
            aligned_eis = alignment(fits_filename)
            print(f"aligned_eis created successfully. Shape: {aligned_eis.data.shape}")
                    
            
            start_date_obj = date_time_obj - timedelta(minutes=60)
            end_date_obj = date_time_obj + timedelta(minutes=120)
            start_date_str = start_date_obj.strftime(DATE_FORMAT)[:-3]
            end_date_str = end_date_obj.strftime(DATE_FORMAT)[:-3]
            goes_downloads = download_goes_data(start_date_str, end_date_str)
    
            
            eis_map = sunpy.map.Map(aligned_eis)
            print(f"eis_map created successfully. Dimensions: {eis_map.dimensions}")    
            
            fig = plt.figure(figsize=(24, 8))
            
            # Plot EIS map
            print('Plotting EIS map')
            ax1 = fig.add_subplot(131, projection=eis_map)
            eis_map.plot(axes=ax1)
            ax1.set_title(f'EIS Map - {filename} ')
            
            # Plot AIA map
            print('Plotting AIA 193 map')
            aia_start = date_time_obj - timedelta(minutes=0.5)
            aia_end = date_time_obj + timedelta(minutes=0.5)
            aia_start_str = aia_start.strftime(DATE_FORMAT)[:-3]
            aia_end_str = aia_end.strftime(DATE_FORMAT)[:-3]
            
            aia_res = Fido.search(a.Time(aia_start_str, aia_end_str), a.Instrument.aia, a.Wavelength(193*u.angstrom))[0][0]
            aia_downloads = Fido.fetch(aia_res, path="./{instrument}/{file}")
            aia_map_plot = sunpy.map.Map(aia_downloads[0])
            
            ax2 = fig.add_subplot(132, projection=aia_map_plot)
            aia_map_plot.plot(axes=ax2)
            
            bottom_left = eis_map.bottom_left_coord
            top_right = eis_map.top_right_coord
            hpc_max = SkyCoord(hpc_max[0]*u.arcsec, hpc_max[1]*u.arcsec, frame = aia_map_plot.coordinate_frame)
            ax2.plot_coord(hpc_max, 'wx', fillstyle='none', markersize=7, color='black')
        
            aia_map_plot.draw_quadrangle(bottom_left, axes=ax2, top_right=top_right, edgecolor="black", linestyle="-", linewidth=2)
            ax2.set_title('AIA Map')
        
            # Plot GOES light curve
            print('Plotting GOES curve')
            goes = ts.TimeSeries(goes_downloads, concatenate=True)
            target_date = datetime.strptime(aligned_eis.meta["date_obs"], '%Y-%m-%dT%H:%M:%S.%f')
            start_str = target_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]
            target_date2 = datetime.strptime(aligned_eis.meta["date_end"], '%Y-%m-%dT%H:%M:%S.%f')
            end_str = target_date2.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]
            
            goes_flare = goes.truncate(start_date_str, end_date_str)
            
            ax3 = fig.add_subplot(133)
            goes.plot(axes=ax3)
            ax3.set_title(f'GOES Light Curve - {goes_class}')
            ax3.set_xlabel('Date Time')
            ax3.set_ylabel('Flux')
            ax3.axvline(x=start_str, color='green')
            ax3.axvline(x=end_str, color='green')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # os.remove(file)
            # os.remove(fits_filename)
            for download in goes_downloads + aia_downloads:
                os.remove(download)
            
            return filename
        except Exception as e:
            print(f"An error occurred while processing {file}: {str(e)}")
            return None
        finally:
            if fig is not None:
                plt.close(fig)        

def load_existing_results(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Start Time", "Peak Time", "End Time", "Class", "hpc_x", "hpc_y"])

def is_row_processed(row, existing_results):
    return not existing_results[(existing_results['Start Time'] == row['Start Time']) & 
                                (existing_results['Peak Time'] == row['Peak Time']) & 
                                (existing_results['End Time'] == row['End Time'])].empty

def create_filtered_flares(flare_df, year):
    eis_xmin = flare_df["EIS xcen"] - 0.5*flare_df["EIS fovx"]
    eis_xmax = flare_df["EIS xcen"] + 0.5*flare_df["EIS fovx"]
    eis_ymin = flare_df["EIS ycen"] - 0.5*flare_df["EIS fovy"]
    eis_ymax = flare_df["EIS ycen"] + 0.5*flare_df["EIS fovy"]
    flare_df["EIS xmin"] = eis_xmin
    flare_df["EIS xmax"] = eis_xmax
    flare_df["EIS ymin"] = eis_ymin
    flare_df["EIS ymax"] = eis_ymax

    correct_indexes = []
    for idx, hpcx in enumerate(flare_df["hpc x"]):
        hpcy = flare_df["hpc y"].iloc[idx]
        if (flare_df["EIS xmin"].iloc[idx] < hpcx < flare_df["EIS xmax"].iloc[idx] and 
            flare_df["EIS ymin"].iloc[idx] < hpcy < flare_df["EIS ymax"].iloc[idx]):
            correct_indexes.append(idx)

    flares = flare_df.iloc[correct_indexes]
    flares.loc[:, "stud_acr"] = flares["stud_acr"].astype(str)
    filtered_flares = flares.loc[~flares["stud_acr"].str.startswith('HH')]

    filtered_flares.to_csv(f'filtered_flares_{year}.csv', sep=',', index=False)
    print(f"Created filtered_flares_{year}.csv")
    return filtered_flares

def get_study_duration(df_study_details, study_acronym):
    # Define a dictionary for fixed durations
    fixed_durations = {
        'DHB_007': 3690,
        'DHB_006': 3690,
        'DHB_007_v2': 3720,
        'DHB_007_v3': 3735,
        'DHB_007_v4': 5535,
        'PRY_loop_footpoints': 1500,
        'JAK_40_AR_STUDY': 1230,
        'JAK_256_AR_STUDY': 7710,
        'EL_DHB_01': 250,
        'EL_DHB_02': 250,
        'EL_DHB_02_v2': 250,
        'Large_CH_Map': 3600,
        'HOP81_new_study': 4050,
        'YKK_ARabund01': 2925,
        'PRY_footpoints_v2': 150,
        'dhb_atlas_120m_30"': 3600,
        'dhb_atlas_30x512': 3600,
        'GDZ_300x384_S2S3_40': 4040,
        'Large_CH_Map_v2': 5400,
        'fullccd_scan_m106': 1060,
        'AR_velocity_map_v2': 150,
        'DHB_006_v2': 7380,
        'EL_FULL_CCD_RASTER': 12300,
        'GDZ_300x360_S2S3_30S': 3030,
        'EL_DHB_01_v2': 375,
        'dhb_atlas_180m_30"': 5400,
        'EL_DHB_01v2': 250,
        'GDZ_AR_HOT_1_280X280': 3500,
        'GDZ_QS1_60x512_60s': 1800,
        'GDZ_PLUME1_2_300_50s': 3750,
        'GDZ_PLUME1_2_300_150': 11250,
        'GDZ_DENS_20x240_ARL2': 1500,
        'FlareResponse03': 400,
        'FlareResponse02': 200,
        'FlareResponse01': 400,
        'FlareResponse01_v2': 640,
        'pry_flare_2': 91.8
    }
    # Check if the study_acronym has a fixed duration
    if study_acronym in fixed_durations:
        return fixed_durations[study_acronym]

    # If not, proceed with dataframe filtering
    filtered_df = df_study_details[df_study_details['acronym'] == study_acronym]
    duration = filtered_df['ra_duration'].values

    if len(duration) != 0:
        return duration[0] / 1000  # return in seconds
    else:
        return None



import sqlite3
import pandas as pd

conn = sqlite3.connect('eis_cat.sqlite')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(eis_raster_db)")
columns = cursor.fetchall()
column_names = [column[1] for column in columns]
cursor.execute("SELECT * FROM eis_raster_db")
rows = cursor.fetchall()
df_study_details = pd.DataFrame(rows, columns=column_names)
conn.close()

@jit(nopython=True)
def check_time_overlap(starts, ends, peaks, time, eis_end_time):
    return np.any((starts < time) & (time < ends) |
                  (starts < eis_end_time) & (eis_end_time < ends) |
                  (time < starts) & (starts < eis_end_time) |
                  (time < ends) & (ends < eis_end_time) |
                  (time < peaks) & (peaks < eis_end_time))

def filter_eis_files(eis_file_dates, eis_study_acros, eis_filenames, df_study_details, starts, ends, peaks, hek_table):
    eis_times = []
    study_durations = []
    flare_eis_filename = []
    start_times = []
    peak_times = []
    end_times = []
    goes_classes = []
    hpc_x = []
    hpc_y = []

    # Convert pandas Timestamps to Unix timestamps (float)
    starts_np = np.array([t.timestamp() for t in starts])
    ends_np = np.array([t.timestamp() for t in ends])
    peaks_np = np.array([t.timestamp() for t in peaks])

    for num, time in tqdm(enumerate(eis_file_dates), total=len(eis_file_dates), desc=f"Filtering relevant EIS files for {YEAR}"):
        if 'PRY_slot' not in str(eis_study_acros[num]) and 'HH' not in str(eis_study_acros[num]):
            study_duration = get_study_duration(df_study_details, str(eis_study_acros[num]))

            if study_duration is not None:
                eis_end_time = time + pd.Timedelta(seconds=study_duration)
                
                # Convert pandas Timestamps to Unix timestamps for Numba function
                time_float = time.timestamp()
                eis_end_time_float = eis_end_time.timestamp()
                
                if check_time_overlap(starts_np, ends_np, peaks_np, time_float, eis_end_time_float):
                    overlap_indices = np.where((starts_np < time_float) & (time_float < ends_np) |
                                               (starts_np < eis_end_time_float) & (eis_end_time_float < ends_np) |
                                               (time_float < starts_np) & (starts_np < eis_end_time_float) |
                                               (time_float < ends_np) & (ends_np < eis_end_time_float) |
                                               (time_float < peaks_np) & (peaks_np < eis_end_time_float))[0]

                    for i in overlap_indices:
                        if time not in eis_times:
                            eis_times.append(time)
                            study_durations.append(study_duration)
                            flare_eis_filename.append(eis_filenames[num])
                            start_times.append(hek_table["Start Time"].iloc[i])
                            peak_times.append(hek_table["Peak Time"].iloc[i])
                            end_times.append(hek_table["End Time"].iloc[i])
                            goes_classes.append(hek_table["GOES Class"].iloc[i])
                            hpc_x.append(hek_table["hpc_x"].iloc[i])
                            hpc_y.append(hek_table["hpc_y"].iloc[i])

    return eis_times, study_durations, flare_eis_filename, start_times, peak_times, end_times, goes_classes, hpc_x, hpc_y
    


# Main Execution
if __name__ == "__main__":
    os.makedirs("output_pics", exist_ok=True)
    for random_num1 in range(10):

        YEARS = range(2011, 2025, 1)
        
        # Constants
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
        for YEAR in YEARS:
            for random_num in range(3):
                print(f"Processing data for year {YEAR}")
                
                hek_table = pd.read_csv(f'./hek_results/hek_results_{YEAR}.csv')
                print(f"Loaded HEK table for {YEAR} with {len(hek_table)} entries")
                
                results_filename = f"filtered_results_{YEAR}.csv"
                print(f"Results will be saved to: {results_filename}")
                filtered_results = load_existing_results(results_filename)
                print(f"Loaded {len(filtered_results)} existing filtered results")
                
                my_df = pd.read_csv('eis_df.csv')
                print(f"Loaded EIS dataframe with {len(my_df)} entries")
                
                starttimes = hek_table["Start Time"]
                endtimes = hek_table["End Time"]
                peaktimes = hek_table["Peak Time"]
                
                filenames = my_df["filename"]
                study_acros = my_df['stud_acr']
                
                eis_file_dates = []
                eis_filenames = []
                eis_study_acros = []
                
                print("Processing EIS filenames...")
                for num, filename in enumerate(filenames):
                    if isinstance(filename, str):
                        try:
                            date_str = filename.split('_')[2][:8]
                            time_str = filename.split('_')[3][:6] 
                            datetime_str = f"{date_str} {time_str}"  
                            datetime_obj = datetime.strptime(datetime_str, '%Y%m%d %H%M%S')
                            eis_file_dates.append(pd.Timestamp(datetime_obj))
                            eis_filenames.append(filename)
                            eis_study_acros.append(study_acros[num])
                        except:
                            pass
                print(f"Processed {len(eis_file_dates)} valid EIS filenames")
                
                start_times = []
                peak_times = []
                end_times = []
                goes_classes = []
                hpc_x = []
                hpc_y = []
                study_durations = []
                flare_eis_filename = []
                eis_times = []
                
                starts = [pd.Timestamp(time) for time in starttimes] # HEK start time of flares
                ends = [pd.Timestamp(time) for time in endtimes] # HEK end time
                peaks = [pd.Timestamp(time) for time in peaktimes] # HEK peak time
                
                print("HEK initial data preparation complete. Ready for further processing.")        

                # do the filtering for hh, slot, time
                eis_times, study_durations, flare_eis_filename, start_times, peak_times, end_times, goes_classes, hpc_x, hpc_y = filter_eis_files(
                    eis_file_dates, eis_study_acros, eis_filenames, df_study_details, starts, ends, peaks, hek_table
                )            
                
                filtered_dates_df = pd.DataFrame({'EIS flare times': eis_times, 'EIS filename': flare_eis_filename,
                'Start Time': start_times, 'Peak Time': peak_times, 'End Time': end_times,
                'Class': goes_classes, 'hpc x': hpc_x, 'hpc y': hpc_y, 'study_durations': study_durations})
    
                # save everything into filtered flares by date csv. At this point some of the flare locations are unknown
                filtered_dates_df.to_csv(f'filtered_flares_by_date_{YEAR}.csv', sep=',', index=False)
            
            
                # Here we try to fill in the flare locations using 94 A difference images
                filtered_hek_by_date = pd.read_csv(f'filtered_flares_by_date_{YEAR}.csv')
            
                for idx, result in tqdm(filtered_hek_by_date.iterrows(), total=len(filtered_hek_by_date), desc="Processing HEK entries"):
                    if is_row_processed(result, filtered_results):
                        print(f"Skipping already processed entry: {result['Start Time']}")
                        continue
            
                    hpc_x, hpc_y = find_flare_location(result) # find flare location using AIA 94
                    if hpc_x is not None and hpc_y is not None: # if there's location
                        filtered_hek_by_date.at[idx, 'hpc_x'] = hpc_x
                        filtered_hek_by_date.at[idx, 'hpc_y'] = hpc_y
                        new_row = {
                            "Start Time": result["Start Time"],
                            "Peak Time": result["Peak Time"],
                            "End Time": result["End Time"],
                            "Class": result["Class"],
                            "hpc_x": hpc_x,
                            "hpc_y": hpc_y,
                        }
                        filtered_results = filtered_results._append(new_row, ignore_index=True)
                        save_dataframe(filtered_results, YEAR)

                # this contains all the flares with locations now
                filtered_results = pd.read_csv(f'filtered_results_{YEAR}.csv')
            
                # # Filter out flares that are not in the EIS data
            
                starttimes = filtered_results["Start Time"]
                endtimes = filtered_results["End Time"]
                filenames = my_df["filename"] # big EIS dataframe
            
                eis_file_dates = []
                eis_filenames = []
            
                for filename in filenames: # every eis data
                    if isinstance(filename, str):
                        try:
                            date_str = filename.split('_')[2][:8]
                            time_str = filename.split('_')[3][:6] 
                            datetime_str = f"{date_str} {time_str}"  
                            datetime_obj = datetime.strptime(datetime_str, '%Y%m%d %H%M%S')
                            eis_file_dates.append(pd.Timestamp(datetime_obj))
                            eis_filenames.append(filename)
                        except:
                            pass
            
                flare_times = []
                flare_eis_filename = []
                goes_classes = []
                hpc_x = []
                hpc_y = []
                eis_times = set()
                
                starts = [pd.Timestamp(time) for time in starttimes]
                ends = [pd.Timestamp(time) for time in endtimes]

                print('file length',len(eis_file_dates))
                print(len(ends))
                
                for num, time in tqdm(enumerate(eis_file_dates)): # for each file
                    if 'PRY_slot' not in str(eis_study_acros[num]) and 'HH' not in str(eis_study_acros[num]):
                        study_duration = get_study_duration(df_study_details, str(eis_study_acros[num]))
                        if study_duration != None:
                            # time is the eis filetime - eis_end_time = calculated end time of eis study
                            eis_end_time = time + pd.Timedelta(seconds=study_duration)
    
                            for i in range(len(starts)):
                                if (starts[i] < time < ends[i] or 
                                    starts[i] < eis_end_time < ends[i] or 
                                    time < starts[i] < eis_end_time or 
                                    time < ends[i] < eis_end_time or 
                                    time < peaks[i] < eis_end_time):
                                    if time not in eis_times:
                                        flare_times.append(time)
                                        flare_eis_filename.append(eis_filenames[num])
                                        goes_classes.append(filtered_results["Class"].iloc[i])
                                        hpc_x.append(filtered_results["hpc_x"].iloc[i])
                                        hpc_y.append(filtered_results["hpc_y"].iloc[i])

                print('hpc length', len(hpc_x))
                xcen, ycen, target, study_id, fovx, fovy, stud_acr, filenames_new = [], [], [], [], [], [], [], []
    
                for file in flare_eis_filename:
                    if file in my_df['filename'].values:
                        matching_row = my_df.loc[my_df['filename'] == file]
                        xcen.extend(matching_row['xcen'].values)
                        ycen.extend(matching_row['ycen'].values)
                        target.extend(matching_row['target'].values)
                        study_id.extend(matching_row['study_id'].values)
                        fovx.extend(matching_row['fovx'].values)
                        fovy.extend(matching_row['fovy'].values)
                        stud_acr.extend(matching_row['stud_acr'].values)
                        filenames_new.append(file.replace("fits.gz", "data.h5").replace("_l0", ""))
                    
            
                flare_df = pd.DataFrame({
                    'study ID': study_id,
                    'flare times': flare_times,
                    'GOES class': goes_classes,
                    'hpc x': hpc_x,
                    'hpc y': hpc_y,
                    'EIS xcen': xcen,
                    'EIS ycen': ycen,
                    'EIS fovx': fovx,
                    'EIS fovy': fovy,
                    'target': target,
                    'stud_acr': stud_acr,
                    'filename': filenames_new
                })
                    
                filtered_flares = create_filtered_flares(flare_df, YEAR)
            
            
                # Process EIS files
                flares = pd.read_csv(f'filtered_flares_{YEAR}.csv')
                filenames = flares['filename']
                parallel_download(filenames)
    
                hpc_xs = flares['hpc x']
                hpc_ys = flares['hpc y']
                goes_classes = flares['GOES class']
                print(len(filenames))
                print(len(hpc_xs))
                for num, file in tqdm(enumerate(filenames)):
                    if os.path.exists(f'./data_eis/{file}'):
                        new_filedir = f'./data_eis/{file}'
                        try:
                            hpc_max = [hpc_xs[num], hpc_ys[num]]
                            process_eis_file(new_filedir, hpc_max, goes_classes[num])
                        except KeyboardInterrupt:
                            print("\nKeyboard interrupt detected. Stopping the process.")
                            break
                        except Exception as e:
                            print(f"An error occurred while processing {file}: {str(e)}")
                    else:
                        pass
                try:
                    for filename in os.listdir('data_eis/'):
                        os.remove(filename)
                except:
                    pass
                print("Processing complete or interrupted.")