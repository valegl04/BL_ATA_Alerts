import pandas as pd
import numpy as np
import os
import scipy
from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.coordinates import Angle
import astropy.constants as const
import time
from datetime import datetime, timezone, timedelta


# Change this to reflect my 4 tunings
FREQ_RANGE = [1336, 2008, 4536, 8336] #MHz

# This file defines some useful tools to keep track of observed targets







# Need to rewrite these two definitions to work with my files.
# Also, currently 0 means unobserved and 1 means observed. Need to make sure they agree
def is_observed(observed_list, source, freq_mhz):

    # Returns 0 if source has been observed at the frequency, otherwise 1

    assert type(source) == str
    assert type(freq_mhz) == int

    source_entries = pd.read_csv(observed_list)
    cfreq_name = "cfreq_%imhz" %freq_mhz

    row = source_entries.loc[source_entries['Plaintext Name'] == source]
    return row[cfreq_name].values[0]

def mark_as_observed(observed_list, source, freq_mhz):

    # Marks the observation as observed by adding 1 to the database entry

    assert type(source) == str
    assert type(freq_mhz) == int

    source_entries = pd.read_csv(observed_list)
    cfreq_name = "cfreq_%imhz" %freq_mhz

    ival = source_entries.loc[source_entries['Plaintext Name'] == source, cfreq_name].values[0]

    if ival == 0:
        source_entries.loc[source_entries['Plaintext Name'] == source, cfreq_name] = 1
        source_entries.to_csv(observed_list, index=False)
    elif ival == 1:
        print("WARNING: has this source been observed before?")








# Defines a function to query Horizons for RA and DEC given a target list DataFrame
def horizons_query(target_list, output_dir, obs_location, obs_start, obs_end, time_step):

    # Query JPL Horizons for RA and DEC for each target in the target_list DataFrame.
    # Saves results to HorizonsOutput.csv in output_dir.
    # Returns a DataFrame with columns: ID, Name, Horizons Name, RA (solsys), DEC (solsys)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("Beginning Horizons query process...")
    print("This is designed to mass query Horizons and receive the current RA/DEC for targets.")
    print(f"Observation location: {obs_location}")
    print(f"Observation start time: {obs_start}")
    print(f"Observation end time: {obs_end}")
    print(f"Time step: {time_step}")

    target_df = pd.read_csv(target_list)
    print("Columns of target_df:")
    print(target_df.columns.tolist())

    name_col = 'Plaintext Name'
    horizons_col = 'Horizons Name'
    flag_col = 'Solar System Flag'
    id_col = 'ID'
    ra_col = 'RA (deg)'
    dec_col = 'DEC'

    results_df = pd.DataFrame(columns=['ID', 'Plaintext Name', 'Horizons Name', 'RA (deg)', 'DEC', 'RA (hours)', 'RA (solsys)', 'DEC (solsys)']) # Creates the empty results dataframe
    # Print the first two rows
    #print(results_df.head(2))

    jpl_targets = target_df[horizons_col]
    Target_ID = target_df[id_col]
    plaintextname = target_df[name_col]

    for idx, target in enumerate(jpl_targets):
        ra_solsys = np.nan
        dec_solsys = np.nan
        # Fills RA/DEC (solsys) with empty values so the process can begin

        if not pd.isna(target):
            # Pull value within parentheses if present, otherwise use as is
            # Some Horizons targets can use just their name, others need an ID code located within parentheses
            if isinstance(target, str) and '(' in target and ')' in target:
                start_idx = target.find('(') + 1
                end_idx = target.find(')', start_idx)
                if end_idx != -1:
                    target_id = target[start_idx:end_idx]
                else:
                    target_id = target
            else:
                target_id = target

            query_start = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
            query_end = (pd.to_datetime(obs_start) + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M')
            print(f"Processing target: {target} | ID: {target_id}")

            try: # Try to execute the query
                # Debug: print the query parameters
                print(f"DEBUG: Query parameters - start: {query_start}, stop: {query_end}, step: {time_step}")
                obj = Horizons(id=target_id, location=obs_location, epochs={'start': query_start, 'stop': query_end, 'step': time_step})
                eph = obj.ephemerides()
                eph = eph[['targetname', 'datetime_str', 'RA', 'DEC']]
                ra_solsys = eph['RA'][0]
                dec_solsys = eph['DEC'][0]
                print(f"{target_id} Queried Horizons successfully at time {query_start} with coordinates RA: {ra_solsys} degrees and DEC: {dec_solsys} degrees.")
            except Exception as e:
                print(f"Error retrieving data for {target_id} for error: {e}")
                # If the error is about TLIST, try with a different step format
                if "TLIST" in str(e) or "no TLIST" in str(e):
                    print(f"TLIST error detected. Attempting alternative query format for {target_id}...")
                    try:
                        # Try with explicit dictionary format
                        obj = Horizons(id=target_id, location=obs_location, 
                                      epochs={'start': query_start, 'stop': query_end, 'step': '60m'})
                        eph = obj.ephemerides()
                        eph = eph[['targetname', 'datetime_str', 'RA', 'DEC']]
                        ra_solsys = eph['RA'][0]
                        dec_solsys = eph['DEC'][0]
                        print(f"SUCCESS (alternative format): {target_id} - RA: {ra_solsys} degrees, DEC: {dec_solsys} degrees.")
                    except Exception as e2:
                        print(f"Alternative query also failed: {e2}")

        # Check if RA or DEC is empty and fill from RA (solsys) and DEC (solsys)
        if pd.isna(target_df[ra_col].iloc[idx]) and not pd.isna(ra_solsys):
            target_df.at[idx, ra_col] = ra_solsys
        if pd.isna(target_df[dec_col].iloc[idx]) and not pd.isna(dec_solsys):
            target_df.at[idx, dec_col] = dec_solsys

        # Convert RA from degrees to hours
        ra_hours = target_df[ra_col].iloc[idx] / 15 if not np.isnan(target_df[ra_col].iloc[idx]) else np.nan

        results_df = pd.concat([results_df, pd.DataFrame({
            'ID': [Target_ID.iloc[idx]],
            'Plaintext Name': [plaintextname.iloc[idx]],
            'Horizons Name': [target],
            'Solar System Flag': [target_df[flag_col].iloc[idx]],
            'RA (deg)': [target_df[ra_col].iloc[idx]],
            'DEC': [target_df[dec_col].iloc[idx]],
            'RA (hours)': [ra_hours]
        })], ignore_index=True)

    # Save the results to CSV after the loop
    results_df.to_csv(os.path.join(output_dir, 'HorizonsOutput.csv'), index=False)
    return results_df

# Example usage:
# results_df = horizons_query(target_list, output_dir, obs_location, obs_start, obs_end, time_step)







# Still need to tweak this
# generate_ephemeris_files creates ephemeris files for a list of targets using JPL Horizons.
# It queries Horizons for each target, interpolates the ephemeris data, and saves the results as .ephem files.
# To use:
# targets_to_process: list/Series of Horizons target names
# id_names_to_process: list/Series of IDs corresponding to targets
# output_dir: directory to save the working ephemeris file
# output_dir2: directory to archive all generated ephemeris files
# scan_start, scan_end: observation time range (strings)
# location: observer location code (string)
# TFMT: time format string (default "%Y-%m-%dT%H:%M")
# TFMT="%Y-%m-%dT%H:%M"

def generate_ephemeris_files(target_to_process, id_to_process, output_dir, output_dir2,
                             scan_start, scan_end, time_step, obs_location):
    print("Generating ephemeris file for the ATA to use:")

    t = target_to_process
    this_id_name = str(id_to_process).replace(" ", "_")
    safe_t = str(t).replace(" ", "_")
    scan_start_name = scan_start.replace(':', '-').replace(' ', 'T')


    if pd.isna(t):  # Skip if target name is NaN
        print(f"Skipping target {t} (ID: {id_to_process}) because it is NaN.")
        return None, None, None, None, None
    else:
        target_id = id_to_process

    print(f"Processing target: {t} | ID: {target_id}")
    #id_types = ['smallbody', 'majorbody', 'astdys', 'mpc', 'designation', 'name', 'alias', 'spacecraft'] # Try all different ID types to keep Horizons happy


    eph = None
    #for id_type in id_types:
    # I commented out the id type try/except. Tab over everything, then uncomment if you want it back.
    try:
        print(f"DEBUG: Ephemeris query parameters - start: {scan_start}, stop: {scan_end}, step: {time_step}")
        obj = Horizons(id=t, location=obs_location,
            epochs={'start': scan_start,
                    'stop':  scan_end,
                    'step':  time_step})
        eph = obj.ephemerides() # Actual execution of query
        print(f"Successfully generated ephemeris for target {t}")
        #print(f"Successfully generated ephemeris using id_type={id_type} for target {t}")
        #break
    except ValueError as e:
        print("Got the value error, trying again...")
        time.sleep(10) #sleep for 10 seconds and try again
        try:
            # Retry once more with slightly different parameters
            print(f"Retrying with alternative step format...")
            obj = Horizons(id=t, location=obs_location,
                epochs={'start': scan_start,
                        'stop':  scan_end,
                        'step':  '60m'})
            eph = obj.ephemerides()
            print(f"Successfully generated ephemeris for target {t} (retry)")
        except Exception as e_retry:
            print(f"Retry also failed: {e_retry}")
    except Exception as e:
        print(f"Exception: {e}")
        if "TLIST" in str(e) or "no TLIST" in str(e):
            print(f"TLIST error detected. Attempting alternative query for {t}...")
            try:
                obj = Horizons(id=t, location=obs_location,
                    epochs={'start': scan_start,
                            'stop':  scan_end,
                            'step':  '60m'})
                eph = obj.ephemerides()
                print(f"Successfully generated ephemeris for target {t} (alternative format)")
            except Exception as e2:
                print(f"Alternative query failed: {e2}")
        #continue
    if eph is None: # This skips empty targets
        print(f"Skipping target {t} (ID: {target_id}) because ephemeris is empty.")
        return None, None, None, None, None


    # Translates time in nanoseconds
    unixTime = (eph['datetime_jd'] - 2440587.5) * 86400
    taiSec = np.int64(np.round((unixTime + 37)*1e9)) # adding leap time in there

    # Pull Az, El, RA, DEC from ephemeris
    az_interp = scipy.interpolate.interp1d(taiSec, eph['AZ'])
    el_interp = scipy.interpolate.interp1d(taiSec, eph['EL'])
    ra_interp = scipy.interpolate.interp1d(taiSec, eph['RA'])
    dec_interp = scipy.interpolate.interp1d(taiSec, eph['DEC'])

    ata_tai_grid = np.arange(taiSec[0], taiSec[-1], 25*1e9)

    az = az_interp(ata_tai_grid) # Outputs to degrees
    el = el_interp(ata_tai_grid) # Outputs to degrees
    ra_deg = ra_interp(ata_tai_grid) # Outputs to degrees
    dec = dec_interp(ata_tai_grid) # Outputs to degrees

    ra = ra_deg / 360 * 24  # Convert RA from degrees to hours

    inv_rad = np.zeros_like(az)
    eph_file_np = np.column_stack([ata_tai_grid, az, el, inv_rad])


    archive_path = os.path.join(output_dir2, f"JPLH_{this_id_name}_{safe_t}_{scan_start_name}.ephem")
    output_ephem = os.path.join(output_dir, f"JPLH_{safe_t}.ephem")
    #generic_ephem = os.path.join(output_dir, "current_ephem.ephem")
    try: # Archive ephemeris
        np.savetxt(archive_path, eph_file_np, fmt='%i   %.5f  %.5f  %.10E')
    except Exception as e:
        print(f"Failed to save archive ephemeris file for {t} at {archive_path} for error: {e}")

    try: # Normal ephemeris output
        np.savetxt(output_ephem, eph_file_np, fmt='%i   %.5f  %.5f  %.10E')
    except Exception as e:
        print(f"Failed to save output ephemeris file for {t} at {output_ephem} for error: {e}")

    print(f"\nEphemeris generated for {scan_start} to {scan_end}.")
    print(f"Target ID: {this_id_name}, Target name: {t}\nSaved Ephemeris as {output_ephem}")

    # Return the RA and DEC
    return az, el, ra, dec, safe_t # RA in hours



# ODS protection from ASP
# def push_ods(ra_deg, dec_deg, length_hours=12):

#     from odsutils import ods_engine
#     from datetime import datetime, timedelta, timezone

#     ods = ods_engine.ODS(conlog='ERROR', defaults="$ods_defaults_ata_B.json")

#     asp_ra_coord_deg  = ra_deg  # RA of ASP at the middle of the night
#     asp_dec_coord_deg = dec_deg # DEC of ASP at the middle of the night

#     #info = check.check_radec(ra_deg*24./360, dec_deg)

#     dt_now = datetime.utcnow()
#     dt_end = dt_now + timedelta(hours=length_hours)

#     #if info['set_time'] < dt_end:
#     #    print(f"source will set before {length_hours}")
#     #    dt_end = info['set_time'] + timedelta(hours = 1) #just do 1 hour after elevation=16

#     cfg = {
#             'src_id': 'ASP',
#             'src_ra_j2000_deg': asp_ra_coord_deg,
#             'src_dec_j2000_deg': asp_dec_coord_deg,
#             'src_start_utc': dt_now.isoformat(),
#             'src_end_utc': dt_end.isoformat()
#         }
#     ods.add(cfg)
#     # ods.update_by_elevation()
#     try:
#         ods.post_ods("/opt/mnt/share/ods_project/ods_exotica.json")
#         ods.assemble_ods("/opt/mnt/share/ods_project", post_to="/opt/mnt/share/ods_upload/ods.json")
#     except:  #anything
#         pass






def sleep_with_continuous_updates_seconds(total_seconds: int):
    remaining_seconds = total_seconds

    while remaining_seconds > 0:
        minutes, seconds = divmod(remaining_seconds, 60)
        print(f"{minutes:02d}:{seconds:02d} remaining", end="\r")
        time.sleep(1)
        remaining_seconds -= 1

    print(f"\rWaited {total_seconds} second{'s' if total_seconds != 1 else ''}")

def countdown_sleep(total_seconds, interval=1):
    # Sleep with periodic countdown display, updates only when minute changes
    remaining = total_seconds
    last_minute = int(remaining / 60) + 1  # Start higher so first minute prints
    while remaining > 0:
        current_minute = int(remaining / 60)
        if current_minute != last_minute:
            print(f"Waiting for ODS... {current_minute}:00 remaining")
            last_minute = current_minute
        time.sleep(min(interval, remaining))
        remaining -= interval
    print("Wait complete!")



def primary_beam_diameter(center_freq):
    # Constants
    c = const.c # Speed of light
    d = 6.1 * u.m # Antenna diameter
    center_freq = center_freq * u.GHz # Assume user input in GHz
    lambda_ = c / center_freq # Wavelength
    # Primary beam diameter
    pb_diam = Angle ((1.22 * lambda_ / d) * u.radian)
    return pb_diam.degree

def synthesized_beam_diameter(center_freq):
    # Synthesized beam diameter
    sb_diam = Angle ((3.5 * 1.2 / center_freq) * u.arcmin) # Assume user input in GHz
    return sb_diam.degree



def push_ods(source, ra_deg, dec_deg, res_start_utc, res_end_utc):

    from odsutils import ods_engine
    from datetime import datetime, timedelta, timezone

    # ods = ods_engine.ODS(output='ERROR')
    # ods.get_defaults('/opt/mnt/share/ods_defaults.json')

    ods = ods_engine.ODS(conlog='ERROR')
    ods.get_defaults('/opt/mnt/share/ods_defaults.json')


    dt_start = res_start_utc.astimezone(timezone.utc)
    dt_end = res_end_utc.astimezone(timezone.utc)

    cfg = {
            'src_id': source,
            'src_ra_j2000_deg': ra_deg,
            'src_dec_j2000_deg': dec_deg,
            'src_start_utc': dt_start.isoformat(),
            'src_end_utc': dt_end.isoformat()
        }
    ods.add(cfg)
    ods.update_by_elevation()
    try:
        ods.post_ods("/opt/mnt/share/ods_upload/ods.json")
        print(f"ODS has been posted for source {source} with the configuration: {cfg}.")
    except Exception as e:
        print(f"ODS has been passed with an exception, it might have failed. Error: {e}")
        pass

