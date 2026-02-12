# CHANGE THESE VALUES FOR YOUR Observation ==========================================================================================================================================================

PROJECT_NAME = 'survey'  # Name of the project. Used in file names and logging
# NOTE: e.g. 'exotica'

# Define scan length
# 300 seconds is 5 minutes
OBS_TIME = 300 #305 #610 #925 # In seconds
# The ATA will automatically add a short buffer.

# The csv_name variable defines the table of targets you want observed. Please ensure it is in the following format:
# NOTE: ID | Plaintext Name | Horizons Name | Solsys Flag | RA (hours) | DEC | RA (deg) | RA (solsys) | DEC (solsys)
# Please see the README for more details.
CSV_NAME = 'survey_targets.csv' # Move this file into the same folder as this script
# NOTE: e.g. 'exotica_targets.csv'

# This file has the ID, Plaintext Name, Solsys Flag, some identifying features if desired (e.g. columns for Sample Type), and columns for each frequency used.
# NOTE: FREQUENCY COLUMNS SHOULD BE NAMED LIKTE: cfreq_1336mhz
OBSERVED_LIST = 'survey_observed.csv'
# NOTE: e.g. 'observed_exotica.csv'

# Here is where you will remove bad antennas from the observation
# Define antennas to remove
REMOVE_ANT = ['1c', '2c']
# NOTE: e.g. REMOVE_ANT = ["2b", "2l", "4e"]
# REMOVE_ANT = [] # For if no antennae need to be removed

# ODS requires a 20 minute lead time before it can activate. If your observation is faster than that time, I would suggest adding the wait
# This will load your targets into the ODS software, then wait for that lead time to complete before beginning the actual observations
QUEUE_WAIT = True # Default is True.

# How often to rebuild the LST-sorted target table during an observation (seconds)
# Should I keep this as seconds or switch to hours?
REORDER_INTERVAL_SECONDS = 3600 # 3600 seconds is 1 hour
# REORDER_INTERVAL_SECONDS = 300 # 600 seconds is 10 minutes, for testing

# Only touch this if you're testing the code. This deactivates the hpguppi recording and the "mark as observed" function
RECORDING = False # For when on a private machine or when on the ATA and you don't want to record. Default is True.
TESTING = True # For when not on the ATA. Default is False.

# RECORDING = True # For when you want to record your observation. This will typically be True.
# TESTING = False # For when you're not testing and you want the ATA to actually observe.

MARKING = False # For marking as observed. Default is True.

# ============================================================================================================================================================================================

# When testing is true or recording is false, ignore the observed list so we still test through the full list.
BYPASS_OBSERVED_CHECK = TESTING or (RECORDING == False)

import atexit
import numpy as np
import scipy
from astropy.constants import c
from astropy.coordinates import Angle, EarthLocation, SkyCoord, AltAz
import astropy.constants as consts
from astropy.time import Time
import astroquery
from astroquery.jplhorizons import Horizons
import astropy

import sys
import time
import argparse
import logging
import csv
import os
from datetime import datetime, timezone, timedelta
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
import traceback

from survey_utils import horizons_query
from survey_utils import is_observed
from survey_utils import mark_as_observed
from survey_utils import generate_ephemeris_files
from survey_utils import primary_beam_diameter
from survey_utils import synthesized_beam_diameter
from survey_utils import push_ods
from survey_utils import sleep_with_continuous_updates_seconds

# NOTE Check that the columns are accurate. They've been changed many times, make sure they're accurate. Then go through and comment which form each thing wants.


if TESTING == True:
    OBS_TIME = 60 # 1 minute observations for testing

print("The following are the user inputs for this observation script:")
print(f"Project Name: {PROJECT_NAME}")
print(f"Observation Time (seconds): {OBS_TIME}")
print(f"CSV Name: {CSV_NAME}")
print("CSV file columns:")
print(pd.read_csv(CSV_NAME).columns.tolist())
print(f"Observed List: {OBSERVED_LIST}")
print(f"Remove Antennas: {REMOVE_ANT}")
print(f"Recording: {RECORDING}")
print(f"Testing Mode: {TESTING}")


if TESTING == True:
    print("TESTING MODE CURRENTLY ACTIVE. ANTENNAE WILL NOT RECORD")


if TESTING == False:
    print("Importing ATA specific functions and beginning script")
    from SNAPobs.snap_hpguppi import snap_hpguppi_defaults as hpguppi_defaults
    from SNAPobs.snap_hpguppi import record_in as hpguppi_record
    from SNAPobs.snap_hpguppi import auxillary as hpguppi_auxillary

    from ATATools import ata_control, logger_defaults
    from SNAPobs import snap_dada, snap_if, snap_config
    from ATATools import ata_sources




output_dir = f"{PROJECT_NAME}_working/" # Working directory for files actively made
os.makedirs(output_dir, exist_ok=True)
output_dir2 = f"{PROJECT_NAME}_archive/" # Archive
os.makedirs(output_dir2, exist_ok=True)
output_dir3 = f"{PROJECT_NAME}_ephemerides/" # Ephemerides
os.makedirs(output_dir3, exist_ok=True)



obs_time_minutes = OBS_TIME / 60  # Convert to minutes
obs_time_hours = OBS_TIME / 3600  # Convert to hours

# ODS lead time in seconds - targets will be pre-loaded this long before observation starts
#ods_lead_time_min = 20 # minutes
ods_lead_time_min = 20 # minutes

if TESTING == True:
    ods_lead_time_min = 1 # 1 minute lead time for testing

ods_lead_time_seconds = ods_lead_time_min * 60 # Convert to seconds

# Calculate how many targets to pre-load to ODS during the lead time
queue_length = -(-ods_lead_time_seconds // OBS_TIME) + 1 # This defines how many targets will be preloaded into ODS

if RECORDING == False or TESTING == True:
    print(F"Testing mode is active (Recording={RECORDING}, Testing={TESTING}), setting ODS queue length to 3 for faster testing.")
    queue_length = 3


print(f"ODS requires {ods_lead_time_min} minutes of lead time to activate. We will wait that long before the first observation.")
print(f"Each target is currently set to observe for {obs_time_minutes:.2f} minutes.")
print(f"Therefore, {queue_length} targets will be pre-loaded to ODS at a time.")


time_step = '1m'  # Time step for the Horizon ephemerides
query_time_step = '6h'  # Time step for the ephemerides. Gives one RA/DEC value for query


# c defined by astropy constants
diameter = 6.1 * u.m #m diameter of ATA
#location = 'Hat Creek Observatory (Allen Array)'
obs_location = {'lon': -121.4733, 'lat': 40.8177, 'elevation': 986}  # Elevation in meters


# Define observing time and location
# Hat Creek Observatory (Allen Array) is 339 in JPL Horizons
#obs_start = '2025-07-01 00:00'
obs_start = datetime.now(timezone.utc).replace(second=0, microsecond=0).strftime('%Y-%m-%d %H:%M')

# Timestamp for the local observation time
hcro_obs_start = datetime.now(timezone.utc).replace(second=0, microsecond=0).astimezone(tz=timezone(timedelta(hours=-8))).strftime('%Y-%m-%d %H:%M')

# Create a filename-safe version of obs_start for use in filenames
obs_start_name = obs_start.replace(':', '-').replace(' ', 'T')
obs_end = (pd.to_datetime(obs_start) + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M') # End time for the observation for Horizons

total_obs_time = pd.to_datetime(obs_end) - pd.to_datetime(obs_start)
print(f"Observation beginning at {obs_start} (Or {hcro_obs_start} local time)")
print(f"Each scan will last {OBS_TIME} seconds ({obs_time_minutes:.3} minutes or {obs_time_hours:.3} hours).")



# Save whole terminal output to a file and archive with obs_start in the name
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_path = os.path.join(output_dir, "terminal_output.txt")
archive_log_path = os.path.join(output_dir2, f"terminal_output_{obs_start_name}.txt")
log_file = open(log_path, "w", encoding="utf-8")
archive_log_file = open(archive_log_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file, archive_log_file)
sys.stderr = Tee(sys.stderr, log_file, archive_log_file)




def build_rearranged_table():
    # Local Sidereal Time
    location = EarthLocation(lat=obs_location['lat'],
                             lon=obs_location['lon'],
                             height=obs_location['elevation'])
    current_start = datetime.now(timezone.utc).replace(second=0, microsecond=0).strftime('%Y-%m-%d %H:%M')
    current_end = (pd.to_datetime(current_start) + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M')
    current_start_name = current_start.replace(':', '-').replace(' ', 'T')

    obs_time_start = Time(current_start)
    lst = obs_time_start.sidereal_time('apparent', longitude=location.lon) # Outputs hours

    # Print LST in hours, degrees, and HH:MM:SS
    print(f"Local Sidereal Time at {current_start} for longitude {location.lon}:")
    print(f"LST (hours): {lst.hour:.4f}")
    print(f"LST (degrees): {lst.deg:.2f}")
    print(f"LST (HH:MM:SS): {lst.to_string(unit=u.hour, sep=':', pad=True, precision=0)}")

    # Query JPL Horizons once for ALL targets
    # Query all Horizons targets within the csv to get their RA/DEC for the time at which you execute
    target_list_df = pd.read_csv(CSV_NAME) # This csv has RA's for Sidereal and Solar targets in the same column, but the Horizons Name column differentiates.
    print("Columns of target list:")
    print(target_list_df.columns.tolist())
    results_df = horizons_query(target_list=CSV_NAME, output_dir=output_dir, obs_location=obs_location,
                                 obs_start=current_start, obs_end=current_end, time_step=query_time_step)
    results_df.to_csv(output_dir + "results_df.csv") # In case you want to save the results to a csv

    # Sort by RA
    ra_sorted_df = results_df.sort_values(by='RA (deg)', ascending=True)
    print("Sorted by RA")
    print("First 3 rows:")
    print(ra_sorted_df.head(3))

    # Somehow this rearranges the tables by LST. Check it to make sure.
    # Use LST to rearrange spreadsheet, where the matching RA is at the top of the list.
    # Find the RA value in sorted dataframe closest to the LST (in hours)
    lst_ra = lst.deg  # LST in hours, same units as RA
    ra_values = ra_sorted_df['RA (hours)'].values

    # Compute absolute difference, ignoring NaNs
    valid_indices = ~pd.isnull(ra_values)
    diffs = np.full_like(ra_values, np.inf, dtype=float)
    diffs[valid_indices] = np.abs(ra_values[valid_indices] - lst_ra)

    # Find the index of the closest RA to LST
    start_idx = np.argmin(diffs)

    # Rearrange the table: start at start_idx, wrap around
    # eg if LST is 20, table will go 20-360, then 0-20 is wrapped onto the bottom
    rearranged_tbl = pd.concat([
        ra_sorted_df.iloc[start_idx:],
        ra_sorted_df.iloc[:start_idx]
    ], ignore_index=True)

    # Save the rearranged table
    print(f"Saving rearranged table around LST: {lst}")

    # Archival copy
    rearranged_filename = f"{PROJECT_NAME}_sources_sorted_LSTstart_{current_start_name}.csv"
    rearranged_tbl.to_csv(os.path.join(output_dir2, rearranged_filename), index=False)

    # Working copy
    rearranged_tbl.to_csv(output_dir + f"{PROJECT_NAME}_sources_sorted_LSTstart.csv", index=False)

    return rearranged_tbl


def main():

    if TESTING == False:
        print("Getting antenna list and removing bad antennas")
        # Get antenna list and separate into 4 tunings
        ant_list = snap_config.get_rfsoc_active_antlist()
        # THIS IS WHERE YOU REMOVE BAD ANTENNAS ========================================================================================================================
        if REMOVE_ANT: # Only process if list is not empty
            for ant in REMOVE_ANT:
                if ant in ant_list:
                    ant_list.remove(ant)

    if TESTING == True:
        print("Testing turned on. This is where bad antennas would be removed.")

    # Can also manually remove antennae if desired
    #ant_list.remove("2j") # As an example
    # THIS IS WHERE YOU REMOVE BAD ANTENNAS ========================================================================================================================

    # Consider making this a prompt or a variable type thing, where you simply type the bad antennas before the observations

    # Bands to Observe and Midpoints:
    # 1.000-1.672 GHz - 1.336 GHz
    # 1.672-2.344 GHz - 2.008 GHz
    # 4.200-4.872 GHz - 4.536 GHz
    # 8.000-8.672 GHZ - 8.336 GHz

    if TESTING == False:
        # Define frequenies like above
        freqs_a = [int(ata_control.get_sky_freq('a'))]*len(ant_list)
        freqs_b = [int(ata_control.get_sky_freq('b'))]*len(ant_list)
        freqs_c = [int(ata_control.get_sky_freq('c'))]*len(ant_list)
        freqs_d = [int(ata_control.get_sky_freq('d'))]*len(ant_list)

        print(f"Center frequencies set to {freqs_a[0]} MHz, {freqs_b[0]} MHz, {freqs_c[0]} MHz, and {freqs_d[0]} MHz.")


        # Register and reserve antennas
        ata_control.reserve_antennas(ant_list)
        atexit.register(ata_control.release_antennas, ant_list, False)

        # Activate SETI nodes (I believe)
        d = {"seti-node%i" %i:[0,1] for i in range(1,15)}
        d_loa = {'seti-node1': [0,1], 'seti-node2': [0,1], 'seti-node3': [0,1], 'seti-node4': [0]}
        d_lob = {'seti-node4': [1], 'seti-node5': [0,1], 'seti-node6': [0,1], 'seti-node7': [0,1]}
        d_loc = {'seti-node8': [0,1], 'seti-node9': [0,1], 'seti-node10': [0,1], 'seti-node11': [0]}
        d_lod = {'seti-node11': [1], 'seti-node12': [0,1], 'seti-node13': [0,1], 'seti-node14': [0,1]}

        # Define beam separation - I currently use the default, which *should* be 5 beamwidths
        f = np.array([freqs_a[0], freqs_b[0], freqs_c[0], freqs_d[0]]) / 1e3 # Put the frequencies into an array, convert to GHz
        # primary_beamwidths = primary_beam_diameter(f)
        synthesized_beamwidths = synthesized_beam_diameter(f)
        separation = 5 * synthesized_beamwidths #separation between on and off beams
        print(f"Beam separation set to {separation} degrees and {synthesized_beamwidths} beamwidths.")

    if TESTING == True:
        print("This is where the antennas would be reserved, the beamwidths would be defined, and the SETI nodes would be activated.")
        freqs_a = [1336]*4
        freqs_b = [2008]*4
        freqs_c = [4536]*4
        freqs_d = [8336]*4

        # assert type(freq_mhz) == int

        # source_entries = pd.read_csv(observed_list)
        # cfreq_name = "cfreq_%imhz" %freq_mhz

        print(f"Center frequencies set to {freqs_a[0]} MHz, {freqs_b[0]} MHz, {freqs_c[0]} MHz, and {freqs_d[0]} MHz for testing.")


    print("==================== Acquiring source list ====================")


    # THIS IS WHERE THE COLUMNS ARE IMPORTANT IN THE CSV                                            ========================

    rearranged_tbl = build_rearranged_table()
    # Read in the needed columns from the csv
    target_df = rearranged_tbl
    plain_names = list(target_df['Plaintext Name'])
    horizons_name = list(target_df['Horizons Name'])
    source_ids = list(target_df['ID'])
    solsys_flag = list(target_df['Solar System Flag'])

    ra_hours_list = list(target_df['RA (hours)']) # RA in hours
    ra_deg_list = list(target_df['RA (deg)']) # RA in degrees
    dec_list = list(target_df['DEC']) # DEC always in degrees

    # Flag to track if this is the first time the queue has been pushed to ODS
    first_queue_pushed = True
    last_reorder_time = time.time()
    print(f"Just acquired source list. The last LST reorder time set to {last_reorder_time} (time.time() function)")

    # Rolling-queue scheduling anchor
    buffer_minutes = 1 # Sets a buffer around each reservation window
    buffer = timedelta(minutes=buffer_minutes)
    schedule_t0 = datetime.now(timezone.utc) # Starting time for the schedule
    first_obs_start = schedule_t0 + timedelta(seconds=ods_lead_time_seconds) + buffer # Wait time for the very first target
    pushed_targets = set()
    observed_targets = set()  # track observed targets in-memory during testing
    schedule_index_by_target = {}  # keep a stable slot index per target

    # Find the first queue_length targets that have not yet been observed
    # If they are in the solar system, query Horizons for their current RA/DEC
    # If they are sidereal, use the RA/DEC from the csv
    # Check altitude and distance from Sun
    # If the target is ready after those things, push ODS.
    # If it's the very first run, wait for the wait_time (30 mins)
    # After that wait time is done (and ONLY for the very first scan), begin observing
    # After that first target finishes, it should go back through the loop and send the next queue_length targets to ODS.
    # It should NOT have the wait time again because it should already be loaded.

    # Step 1: Begin the loop itself, starting with the top of the LST arranged table
    i = 0
    while i < len(plain_names):
        elapsed_since_reorder = time.time() - last_reorder_time
        seconds_until_reorder = max(0, REORDER_INTERVAL_SECONDS - elapsed_since_reorder)
        print(
            f"Time since last LST reorder: {elapsed_since_reorder:.0f}s | "
            f"Time until next reorder: {seconds_until_reorder:.0f}s"
        )
        if time.time() - last_reorder_time >= REORDER_INTERVAL_SECONDS:
            current_source = plain_names[i] if i < len(plain_names) else None
            print("Rebuilding LST-sorted target table (hourly refresh).")
            rearranged_tbl = build_rearranged_table()
            target_df = rearranged_tbl
            plain_names = list(target_df['Plaintext Name'])
            horizons_name = list(target_df['Horizons Name'])
            source_ids = list(target_df['ID'])
            solsys_flag = list(target_df['Solar System Flag'])
            ra_hours_list = list(target_df['RA (hours)']) # RA in hours
            ra_deg_list = list(target_df['RA (deg)']) # RA in degrees
            dec_list = list(target_df['DEC']) # DEC always in degrees

            if current_source in plain_names:
                i = plain_names.index(current_source)
            else:
                i = 0

            last_reorder_time = time.time()

        source = plain_names[i]
        # Skip if already observed in this run (testing/no marking)
        if (BYPASS_OBSERVED_CHECK or RECORDING == False) and source in observed_targets:
            print(f"Target {source} already observed in this run. Skipping.")
            i += 1
            continue
        # Redefine the time and start of the loop for each iteration
        scan_start = datetime.now(timezone.utc).replace(second=0, microsecond=0).strftime('%Y-%m-%d %H:%M')
        scan_end = (pd.to_datetime(scan_start) + timedelta(minutes=60)).strftime('%Y-%m-%d %H:%M')
        ata_scan_start = datetime.now(timezone.utc).replace(second=0, microsecond=0).astimezone(tz=timezone(timedelta(hours=-8))).strftime('%Y-%m-%d %H:%M')
        print("\n==================== Top of the loop ====================")
        print(f"Current time = {scan_start} UTC (Or {ata_scan_start} local time)")
        print(f"Current target: {i} - {source}\n")


        # Step 2: Find next queue_length targets for queuing to ODS
        targets_to_queue = []
        queue_idx = i
        queue_count = 0
        while queue_idx < len(plain_names) and queue_count < queue_length:
            queue_source = plain_names[queue_idx]
            print(f"Beginning queue check for target: {queue_source}")

            # Step 2: Check if this target has been previously observed
            if BYPASS_OBSERVED_CHECK or (
                is_observed(OBSERVED_LIST, queue_source, freqs_a[0]) == 0 or
                is_observed(OBSERVED_LIST, queue_source, freqs_b[0]) == 0 or
                is_observed(OBSERVED_LIST, queue_source, freqs_c[0]) == 0 or
                is_observed(OBSERVED_LIST, queue_source, freqs_d[0]) == 0
            ):
                print(f"Target {queue_source} has not yet been observed.")

                # Get RA/DEC for this target
                queue_ra_deg = None
                queue_dec = None

                if solsys_flag[queue_idx] == 1:
                    this_horizons_name = horizons_name[queue_idx]
                    print(f"Target {queue_source} is a Solar System target ({this_horizons_name}). Querying Horizons for RA/DEC.")
                    if isinstance(this_horizons_name, str) and '(' in this_horizons_name and ')' in this_horizons_name:
                        start_idx = this_horizons_name.find('(') + 1
                        end_idx = this_horizons_name.find(')', start_idx)
                        if end_idx != -1:
                            queue_name_to_query = this_horizons_name[start_idx:end_idx]
                            try:
                                obj = Horizons(id=queue_name_to_query, location=obs_location, epochs={'start': scan_start, 'stop': scan_end, 'step': '60m'})
                                eph = obj.ephemerides()
                                queue_ra_deg = float(eph['RA'][0])
                                queue_dec = float(eph['DEC'][0])
                            except Exception as e:
                                print(f"Error querying Horizons for {queue_source}: {e}. Skipping.")
                                queue_idx += 1
                                continue
                else:
                    print(f"Target {queue_source} is a Sidereal target. Using RA/DEC from CSV.")
                    queue_ra_deg = ra_deg_list[queue_idx]
                    queue_dec = dec_list[queue_idx]

                if queue_ra_deg is None or queue_dec is None:
                    print(f"Could not determine RA/DEC for {queue_source}. Skipping.")
                    queue_idx += 1
                    continue

                # Check altitude
                observing_location = EarthLocation(lat='40.8177', lon='-121.4733', height=986 * u.m)
                observing_time = Time(scan_start)
                aa = AltAz(location=observing_location, obstime=observing_time)
                coord = SkyCoord(ra=queue_ra_deg * u.deg, dec=queue_dec * u.deg)
                altaz_coord = coord.transform_to(aa)
                queue_alt = altaz_coord.alt.value

                if not (85 > queue_alt > 21):
                    print(f"Target {queue_source} is out of altitude bounds ({queue_alt:.1f} degrees). Skipping.")
                    queue_idx += 1
                    continue



                # Check Sun separation
                if TESTING == False:
                    sun = ata_sources.check_source("sun")

                if TESTING == True:
                    try:
                        obj = Horizons(id="10", location=obs_location, epochs=Time(scan_start).jd)
                        eph = obj.ephemerides()
                        sun = {"ra": float(eph["RA"][0]), "dec": float(eph["DEC"][0])} # Outputs RA in degrees
                        print("Testing mode: Sun position queried from JPL Horizons.")
                    except Exception as e:
                        print(f"Testing mode: JPL Horizons query for Sun failed: {e}.")
                sun_ra = sun['ra']
                sun_dec = sun['dec']
                target_coord = SkyCoord(ra=queue_ra_deg * u.deg, dec=queue_dec * u.deg)
                sun_coord = SkyCoord(ra=sun_ra * u.deg, dec=sun_dec * u.deg)
                separation_from_sun = target_coord.separation(sun_coord).deg
                print(f"Current separation from Sun for target {queue_source}: {separation_from_sun:.2f} degrees")

                if separation_from_sun < 10:
                    print(f"Target {queue_source} is too close to the Sun ({separation_from_sun:.2f} degrees < 10). Skipping.")
                    queue_idx += 1
                    continue



                # Check Moon separation
                if TESTING == False:
                    moon = ata_sources.check_source("moon")

                if TESTING == True:
                    try:
                        obj = Horizons(id="301", location=obs_location, epochs=Time(scan_start).jd)
                        eph = obj.ephemerides()
                        moon = {"ra": float(eph["RA"][0]), "dec": float(eph["DEC"][0])}  # Outputs RA in degrees
                        print("Testing mode: Moon position queried from JPL Horizons.")
                    except Exception as e:
                        print(f"Testing mode: JPL Horizons query for Moon failed: {e}.")
                moon_ra = moon['ra']
                moon_dec = moon['dec']
                target_coord = SkyCoord(ra=queue_ra_deg * u.deg, dec=queue_dec * u.deg)
                moon_coord = SkyCoord(ra=moon_ra * u.deg, dec=moon_dec * u.deg)
                separation_from_moon = target_coord.separation(moon_coord).deg
                print(f"Current separation from Moon for target {queue_source}: {separation_from_moon:.2f} degrees")

                if separation_from_moon < 10:
                    print(f"Target {queue_source} is too close to the Moon ({separation_from_moon:.2f} degrees < 10). Skipping.")
                    queue_idx += 1
                    continue

                # Count it toward the queue window
                queue_count += 1

                # Only push if this target has not already been pushed
                if queue_source not in pushed_targets:
                    targets_to_queue.append((queue_idx, queue_source, queue_ra_deg, queue_dec))
                    print(f"Target {queue_source} added to queue (new push).")
                else:
                    print(f"Target {queue_source} already pushed earlier; skipping duplicate push.")

                if queue_count >= queue_length:
                    print(f"Reached queue length of {queue_length}. Proceeding with ODS push function.")
                    break

            queue_idx += 1

        if len(targets_to_queue) < queue_length:
            print(f"Queue fill incomplete: only {len(targets_to_queue)}/{queue_length} targets met criteria after scanning the remaining list.")

        # Push all queued targets to ODS (outside inner loop, after targets are evaluated)
        if targets_to_queue:
            print(f"Pushing {len(targets_to_queue)} queued target(s) to ODS...")
            print(f"Queued targets are: {[t[1] for t in targets_to_queue]}")

            for queue_idx, queued_target, queue_ra_deg, queue_dec in targets_to_queue:
                # Keep a stable schedule index for each target
                if queued_target not in schedule_index_by_target:
                    schedule_index_by_target[queued_target] = queue_idx
                slot_index = schedule_index_by_target[queued_target]

                obs_start_utc = first_obs_start + timedelta(seconds=OBS_TIME * slot_index)
                obs_end_utc = obs_start_utc + timedelta(seconds=OBS_TIME)

                # Apply buffer to reservation window
                res_start_utc = obs_start_utc - buffer
                res_end_utc = obs_end_utc + buffer

                if RECORDING == True:
                    print(f"Pushing {queued_target} at coords ({queue_ra_deg} RA, {queue_dec} Dec) to ODS from {res_start_utc.isoformat()} to {res_end_utc.isoformat()}")
                    push_ods(queued_target, queue_ra_deg, queue_dec, res_start_utc, res_end_utc)
                    print(f"Successfully pushed {queued_target} to ODS.")
                else:
                    print(f"Testing mode: would have pushed {queued_target} at coords ({queue_ra_deg} RA, {queue_dec} Dec) to ODS from {res_start_utc.isoformat()} to {res_end_utc.isoformat()}")

                pushed_targets.add(queued_target)
                print(f"Pushed {queued_target} to ODS queue")

            # Add sleep only on the first queue push
            if first_queue_pushed:

                if TESTING == True:
                    print("Testing mode active, doing a shorter wait for the first ODS push.")
                    first_queue_pushed = False
                    if QUEUE_WAIT == True:
                        sleep_with_continuous_updates_seconds(60) # 1 minute wait for testing

                if TESTING == False:
                    print(f"First queue pushed to ODS. Sleeping for {ods_lead_time_min} minutes to allow ODS to activate...")
                    if QUEUE_WAIT == True:
                        sleep_with_continuous_updates_seconds(ods_lead_time_seconds)
                    first_queue_pushed = False
                    print("Done waiting for the first ODS activation.")

            print("All queued targets pushed to ODS. Beginning actual observations")


        # Step 3: Perform the actual observation
        # Now that the queue has been pushed to ODS, we can do the actual observation with the step by step ephemerides
        # Make sure that the queue deals with the first queue_length targets from the current index within the greater for loop, but that the observation only takes place with the current index target


        # If solar system flag = 1, use Horizons name and generate_ephemeris_files function.
        # ------------------------------------------------------------------------------------------------------------------------------
        if solsys_flag[i] == 1:

            print(f"Target {source} is a Solar System target. Generating ephemeris files.")
            # Generate ephemeris files

            this_horizons_name = horizons_name[i] # List to index for loop
            # Pull Horizons specific ID from within parentheses in Horizons name
            if isinstance(this_horizons_name, str) and '(' in this_horizons_name and ')' in this_horizons_name:
                start_idx = this_horizons_name.find('(') + 1
                end_idx = this_horizons_name.find(')', start_idx)
                if end_idx != -1:
                    name_to_query = this_horizons_name[start_idx:end_idx]
                else:
                    name_to_query = this_horizons_name
            else:
                name_to_query = this_horizons_name

            print(f"Querying Horizons for target {source} (ID: {source_ids[i]}) with name {name_to_query}")

            az, el, ra, dec, safe_t = generate_ephemeris_files(target_to_process=name_to_query, id_to_process=source_ids[i],
                                    output_dir=output_dir3, output_dir2=output_dir2,
                                scan_start=scan_start, scan_end=scan_end, time_step=time_step, obs_location=obs_location)
            # Generate_ephemeris_files gives RA in hours. RA is originally queried in degrees, then the function converts.

            if ra is None or dec is None:
                print(f"Error: Could not retrieve RA and DEC for {source} (ID: {source_ids[i]}). Skipping this target.")
                i += 1
                continue
            # Output_dir3 is for ephemerides, output_dir2 is for archive
            print(f"Ephemeris files for {source} (Queried with {name_to_query}) generated successfully.")

            # Pull first values from RA and DEC tables
            ra_hours = ra[0] # In hours
            dec = dec[0]

            az = az[0] # In degrees
            el = el[0] # In degrees

            ra_deg = ra_hours * 360 / 24  # Convert RA from hours to degrees

            # Check that the source is at an appropriate elevation
            observing_location = EarthLocation(lat='40.8177', lon='-121.4733', height=986 * u.m)
            observing_time = Time(scan_start)
            aa = AltAz(location=observing_location, obstime=observing_time)
            coord = SkyCoord(ra_deg * u.deg, dec * u.deg) # Needs RA in degrees
            altaz_coord = coord.transform_to(aa)
            alt = altaz_coord.alt.value


        # ------------------------------------------------------------------------------------------------------------------------------
        if solsys_flag[i] == 0:
            # If not a solar system target, use RA and DEC from CSV
            print(f"Target {source} is not a Solar System target. Using RA and DEC from CSV.")
            ra_hours = ra_hours_list[i] # In hours
            ra_deg = ra_deg_list[i] # In degrees
            dec = dec_list[i]


            # Check that the source is at an appropriate elevation
            observing_location = EarthLocation(lat='40.8177', lon='-121.4733', height=986 * u.m)
            observing_time = Time(scan_start)
            aa = AltAz(location=observing_location, obstime=observing_time)
            coord = SkyCoord(ra_deg * u.deg, dec * u.deg) # Needs RA in degrees
            altaz_coord = coord.transform_to(aa)
            alt = altaz_coord.alt.value



        print(f"Current Altitude: {alt} degrees")
        # Need to check this. Make sure it's pulling RA/DEC from only Solsys OR Sidereal, not both
        if 85 > alt > 21:
            print(f"Altitude is within bounds, tracking source {source}")
        else:
            print(f"{source} is not within elevation bounds ({alt}), so is being skipped.")
            i += 1
            continue



        # Sun and Moon protection
        # Check Sun separation
        if TESTING == False:
            sun = ata_sources.check_source("sun")

        if TESTING == True:
            try:
                obj = Horizons(id="10", location=obs_location, epochs=Time(scan_start).jd)
                eph = obj.ephemerides()
                sun = {"ra": float(eph["RA"][0]), "dec": float(eph["DEC"][0])} # Outputs RA in degrees
                print("Testing mode: Sun position queried from JPL Horizons.")
            except Exception as e:
                print(f"Testing mode: JPL Horizons query for Sun failed: {e}.")
        sun_ra = sun['ra']
        sun_dec = sun['dec']
        target_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec * u.deg)
        sun_coord = SkyCoord(ra=sun_ra * u.deg, dec=sun_dec * u.deg)
        separation_from_sun = target_coord.separation(sun_coord).deg
        print(f"Current separation from Sun for target {source}: {separation_from_sun:.2f} degrees")

        if separation_from_sun < 10:
            print(f"Target {source} is too close to the Sun ({separation_from_sun:.2f} degrees < 10). Skipping.")
            queue_idx += 1
            continue



        # Check Moon separation
        if TESTING == False:
            moon = ata_sources.check_source("moon")

        if TESTING == True:
            try:
                obj = Horizons(id="301", location=obs_location, epochs=Time(scan_start).jd)
                eph = obj.ephemerides()
                moon = {"ra": float(eph["RA"][0]), "dec": float(eph["DEC"][0])}  # Outputs RA in degrees
                print("Testing mode: Moon position queried from JPL Horizons.")
            except Exception as e:
                print(f"Testing mode: JPL Horizons query for Moon failed: {e}.")
        moon_ra = moon['ra']
        moon_dec = moon['dec']
        target_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec * u.deg)
        moon_coord = SkyCoord(ra=moon_ra * u.deg, dec=moon_dec * u.deg)
        separation_from_moon = target_coord.separation(moon_coord).deg
        print(f"Current separation from Moon for target {source}: {separation_from_moon:.2f} degrees")

        if separation_from_moon < 10:
            print(f"Target {source} is too close to the Moon ({separation_from_moon:.2f} degrees < 10). Skipping.")
            queue_idx += 1
            continue




        if TESTING == False:

            print("Beginning tracking function based on solar system flag.")

            if solsys_flag[i] == 1:
            # Track source
                print("Target is within the solar system, tracking using Horizons Ephemeris.")
                print(f"Tracking source {source}...")
                eph_id = ata_control.upload_ephemeris(f"../{PROJECT_NAME}_ephemerides/JPLH_{safe_t}.ephem")
                for ant in ant_list[:-1]:
                    ata_control.track_ephemeris(eph_id, [ant], wait=False)
                ata_control.track_ephemeris(eph_id, [ant_list[-1]], wait=True)
            if solsys_flag[i] == 0:
                print("Target is Sidereal, tracking using given RA/DEC.")
                print(f"Tracking source {source}...")
                ata_control.make_and_track_ra_dec(ra_hours, dec, ant_list) # Wants RA Hours

        if TESTING == True:
            print("Testing mode active, skipping actual tracking function.")
            if solsys_flag[i] == 1:
                print("Target is within the solar system, would be tracking using Horizons Ephemeris.")
            if solsys_flag[i] == 0:
                print("Target is Sidereal, would be tracking using given RA/DEC.")





        if RECORDING == True:

            # BEGIN HPGUPPI RECORDING
            # PUT THIS BEHIND IF TESTING == FALSE FLAG
            # Off beams need RA in HOURS
            # Define off beams
            dec1 = dec + separation
            keyval_dict_loa = {'RA_OFF0': ra, 'DEC_OFF0': dec,
                    'RA_OFF1': ra, 'DEC_OFF1': dec1[0]}
            keyval_dict_lob = {'RA_OFF0': ra, 'DEC_OFF0': dec,
                    'RA_OFF1': ra, 'DEC_OFF1': dec1[1]}
            keyval_dict_loc = {'RA_OFF0': ra, 'DEC_OFF0': dec,
                    'RA_OFF1': ra, 'DEC_OFF1': dec1[2]}
            keyval_dict_lod = {'RA_OFF0': ra, 'DEC_OFF0': dec,
                    'RA_OFF1': ra, 'DEC_OFF1': dec1[3]}

            #keyval_dict = {'RA_OFF0': ra, 'DEC_OFF0': dec}
            hpguppi_auxillary.publish_keyval_dict_to_redis(
                    keyval_dict_loa, d_loa, postproc=False)
            hpguppi_auxillary.publish_keyval_dict_to_redis(
                    keyval_dict_lob, d_lob, postproc=False)
            hpguppi_auxillary.publish_keyval_dict_to_redis(
                    keyval_dict_loc, d_loc, postproc=False)
            hpguppi_auxillary.publish_keyval_dict_to_redis(
                    keyval_dict_lod, d_lod, postproc=False)

            time.sleep(20)

            obs_start_in = 10

            print("REMOVED POST PROCESSING BLOCK, USE AT OWN RISK")
            print("Sleeping for 60 seconds instead...")
            #hpguppi_record.block_until_post_processing_waiting(d)
            #print('Post proc is done for\n', d)
            sleep_with_continuous_updates_seconds(60)

            # Begin new observation of target
            print(f"\nStarting new observation of {source}")
            #print("Remember that you can type 'stop' and press Enter to end the loop after the current scan.")
            hpguppi_record.record_in(obs_start_in, OBS_TIME + 5, hashpipe_targets = d) # ATA recording function - Actual observation
            # Start recording -- record_in does NOT block
            print("\n")
            if solsys_flag[i] == 1:
                print(f"Source {source} is in the solar system. Currently at RA: {ra_hours} hours ({ra_deg} degrees) and DEC: {dec} degrees.")
            if solsys_flag[i] == 0:
                print(f"Source {source} is Sidereal with RA: {ra_hours} hours ({ra_deg} degrees) and DEC: {dec} degrees.")
            print(f"========================= Recording of {source} (index {i}) for {OBS_TIME} seconds started =========================")
            sleep_with_continuous_updates_seconds(OBS_TIME + obs_start_in + 10)
            #time.sleep(OBS_TIME + obs_start_in + 5)

            print(f"Observation of {source} (ID: {source_ids[i]}) completed")
            time.sleep(5) # A little buffer to make sure it really stopped observing.


            # I kept this separated just in case. Redundant, but doesn't really matter
            # Can rename later so we can observe and record, but not mark as observed.
            if MARKING == True:
                print("Marking as observed")
                freqs = [freqs_a[0], freqs_b[0], freqs_c[0], freqs_d[0]]
                for freq in freqs:
                    mark_as_observed(OBSERVED_LIST, source, freq)
                    print(f"{source} has been successfully marked as observed at {freq}MHz")

            if TESTING == True:
                print("Temporarilty removed marking as observed due to testing mode.")

            print(f"Observation cycle complete for {source}. If you'd like to end this session, please do so now.")
            print("============================================================\n")
            sleep_with_continuous_updates_seconds(30)

        if RECORDING == False:
            print("Recording disabled. This will not record and will not mark as observed.")
            print("This is where observation would have taken place.")
            print(f"Observation cycle is complete for {source}. If you'd like to end this session, please do so now.")
            observed_targets.add(source) # Remember it was observed this run
            sleep_with_continuous_updates_seconds(30)



    else:
        print(f'Target {source} has already been observed, skipping to the next.')

    i += 1



if __name__ == "__main__":
    main()

