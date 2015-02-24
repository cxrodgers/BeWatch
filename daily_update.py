"""Module for running daily updates on database"""
import os
import numpy as np
import glob
import pandas
import ArduFSM
import my
from ArduFSM import TrialMatrix, TrialSpeak, mainloop


def daily_update():
    """Update the databases with current behavior and video files
    
    This should be run on marvin locale.
    """
    if LOCALE != 'marvin':
        raise ValueError("this must be run on marvin")
    
    daily_update_behavior()
    daily_update_video()
    daily_update_overlap_behavior_and_video()
    daily_update_trial_matrix()
    daily_update_perf_metrics()

def daily_update_behavior():
    """Update behavior database"""
    # load
    behavior_files_df = search_for_behavior_files(
        behavior_dir=PATHS['behavior_dir'],
        clean=True)
    
    # store copy for error check
    behavior_files_df_local = behavior_files_df.copy()
    
    # locale-ify
    behavior_files_df['filename'] = behavior_files_df['filename'].str.replace(
        PATHS['behavior_dir'], '$behavior_dir$')
    
    # save
    filename = os.path.join(PATHS['database_root'], 'behavior.csv')
    behavior_files_df.to_csv(filename, index=False)
    
    # Test the reading/writing is working
    bdf = get_behavior_df()
    if not (behavior_files_df_local == bdf).all().all():
        raise ValueError("read/write error in behavior database")
    
def daily_update_video():
    """Update video database"""
    # find video files
    video_files = glob.glob(os.path.join(PATHS['video_dir'], '*.mp4'))
    
    # TODO: load existing video files and use as cache
    # TODO: error check here; if no videos; do not trash cache
    
    # Parse into df
    video_files_df = parse_video_filenames(video_files, verbose=False,
        cached_video_files_df=None)

    # store copy for error check
    video_files_df_local = video_files_df.copy()

    # locale-ify
    video_files_df['filename'] = video_files_df['filename'].str.replace(
        PATHS['video_dir'], '$video_dir$')
    
    # Save
    filename = os.path.join(PATHS['database_root'], 'video.csv')
    video_files_df.to_csv(filename, index=False)    
    
    # Test the reading/writing is working
    vdf = get_video_df()
    if not (video_files_df_local == vdf).all().all():
        raise ValueError("read/write error in video database")    

def daily_update_overlap_behavior_and_video():
    """Update the linkage betweeen behavior and video df
    
    Should run daily_update_behavior and daily_update_video first
    """
    # Load the databases
    behavior_files_df = get_behavior_df()
    video_files_df = get_video_df()

    # Find the best overlap
    new_behavior_files_df = find_best_overlap_video(
        behavior_files_df, video_files_df)
    
    # Join video info
    joined = new_behavior_files_df.join(video_files_df, 
        on='best_video_index', rsuffix='_video')
    
    # Drop on unmatched
    joined = joined.dropna()
    
    # Add the delta-time guess
    # Negative timedeltas aren't handled by to_timedelta in the loading function
    # So store as seconds here
    guess = joined['dt_start_video'] - joined['dt_start']
    joined['guess_vvsb_start'] = guess / np.timedelta64(1, 's')
    
    # locale-ify
    joined['filename'] = joined['filename'].str.replace(
        PATHS['behavior_dir'], '$behavior_dir$')    
    joined['filename_video'] = joined['filename_video'].str.replace(
        PATHS['video_dir'], '$video_dir$')    
        
    # Save
    filename = os.path.join(PATHS['database_root'], 'behave_and_video.csv')
    joined.to_csv(filename, index=False)

def daily_update_trial_matrix(start_date=None, verbose=False):
    """Cache the trial matrix for every session
    
    TODO: use cache
    """
    # Get
    behavior_files_df = get_behavior_df()
    
    # Filter by those after start date
    behavior_files_df = behavior_files_df[ 
        behavior_files_df.dt_start >= start_date]
    
    # Calculate trial_matrix for each
    session2trial_matrix = {}
    for irow, row in behavior_files_df.iterrows():
        # Check if it already exists
        filename = os.path.join(PATHS['database_root'], 'trial_matrix', 
            row['session'])
        if os.path.exists(filename):
            continue

        if verbose:
            print filename

        # Otherwise make it
        trial_matrix = TrialMatrix.make_trial_matrix_from_file(row['filename'])
        
        # And store it
        trial_matrix.to_csv(filename)

def daily_update_perf_metrics(start_date=None, verbose=False):
    """Calculate simple perf metrics for anything that needs it.
    
    start_date : if not None, ignores all behavior files before this date
        You can also pass a string like '20150120'
    
    This assumes trial matrices have been cached for all sessions in bdf.
    Should error check for this.
    
    To add: percentage of forced trials. EV of various biases instead
    of FEV
    """
    # Get
    behavior_files_df = get_behavior_df()

    # Filter by those after start date
    behavior_files_df = behavior_files_df[ 
        behavior_files_df.dt_start >= start_date]

    # Load what we've already calculated
    pmdf = get_perf_metrics()

    # Calculate any that need it
    new_pmdf_rows_l = []
    for idx, brow in behavior_files_df.iterrows():
        # Check if it already exists
        session = brow['session']
        if session in pmdf['session'].values:
            if verbose:
                print "skipping", session
            continue
        
        # Otherwise run
        trial_matrix = get_trial_matrix(session)
        metrics = calculate_perf_metrics(trial_matrix)
        
        # Store
        metrics['session'] = session
        new_pmdf_rows_l.append(metrics)
    
    # Join on the existing pmdf
    new_pmdf_rows = pandas.DataFrame.from_records(new_pmdf_rows_l)
    new_pmdf = pandas.concat([pmdf, new_pmdf_rows],
        verify_integrity=True,
        ignore_index=True)
    
    # Columns are sorted after concatting
    # Re-use original, this should be specified somewhere though
    if new_pmdf.shape[1] != pmdf.shape[1]:
        raise ValueError("missing/extra columns in perf metrics")
    new_pmdf = new_pmdf[pmdf.columns]
    
    # Save
    filename = os.path.join(PATHS['database_root'], 'perf_metrics.csv')
    new_pmdf.to_csv(filename, index=False)

