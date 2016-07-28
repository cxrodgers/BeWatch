"""Module for running daily updates on database"""
import os
import numpy as np
import glob
import pandas
import ArduFSM
import my
import BeWatch
from ArduFSM import TrialMatrix, TrialSpeak, mainloop


def daily_update():
    """Update the databases with current behavior and video files
    
    This should be run on marvin locale.
    """
    if BeWatch.db.get_locale() != 'marvin':
        raise ValueError("this must be run on marvin")
    
    daily_update_behavior()
    daily_update_video()
    daily_update_overlap_behavior_and_video()
    daily_update_trial_matrix()
    daily_update_perf_metrics()

def daily_update_behavior():
    """Update behavior database"""
    # load the current database
    current_bdf = BeWatch.db.get_behavior_df()

    # get new records
    PATHS = BeWatch.db.get_paths()
    newly_added_bdf = BeWatch.db.search_for_behavior_files(
        behavior_dir=PATHS['behavior_dir'],
        clean=True)
    
    # concatenate all existing records with all records previously in
    # the database. for duplicates, keep the newly processed version
    concatted = pandas.concat([current_bdf, newly_added_bdf],
        ignore_index=True, verify_integrity=True)
    new_bdf = concatted.drop_duplicates(subset='session',
        take_last=True).reset_index(drop=True)

    # store copy for error check
    new_bdf_copy = new_bdf.copy()
    
    # locale-ify
    new_bdf['filename'] = new_bdf['filename'].str.replace(
        PATHS['behavior_dir'], '$behavior_dir$')
    
    # save
    filename = os.path.join(PATHS['database_root'], 'behavior.csv')
    new_bdf.to_csv(filename, index=False)
    
    # Test the reading/writing is working
    bdf_reloaded = BeWatch.db.get_behavior_df()
    if not (new_bdf_copy == bdf_reloaded).all().all():
        raise ValueError("read/write error in behavior database")
    
def daily_update_video():
    """Update video database
    
    Finds video files in PATHS['video_dir']
    Extracts timing information from them
    Updates video.csv on disk.
    """
    PATHS = BeWatch.db.get_paths()
    # find video files
    mp4_files = glob.glob(os.path.join(PATHS['video_dir'], '*.mp4'))
    mkv_files = glob.glob(os.path.join(PATHS['video_dir'], '*.mkv'))
    video_files = mp4_files + mkv_files
    
    # Load existing video file dataframe and use as a cache
    # This way we don't have to reprocess videos we already know about
    vdf = BeWatch.db.get_video_df()    
    
    # Parse into df
    video_files_df = BeWatch.db.parse_video_filenames(
        video_files, verbose=True,
        cached_video_files_df=vdf)

    # store copy for error check (to ensure that localeifying and
    # writing to disk didn't corrupt anything)
    video_files_df_local = video_files_df.copy()

    # locale-ify
    video_files_df['filename'] = video_files_df['filename'].str.replace(
        PATHS['video_dir'], '$video_dir$')
    
    # Save
    filename = os.path.join(PATHS['database_root'], 'video.csv')
    video_files_df.to_csv(filename, index=False)    
    
    # Test the reading/writing is working
    # Although if it failed, it's too late
    vdf = BeWatch.db.get_video_df()
    if not (video_files_df_local == vdf).all().all():
        raise ValueError("read/write error in video database")    

def daily_update_overlap_behavior_and_video():
    """Update the linkage betweeen behavior and video df
    
    Should run daily_update_behavior and daily_update_video first
    """
    PATHS = BeWatch.db.get_paths()
    # Load the databases
    behavior_files_df = BeWatch.db.get_behavior_df()
    video_files_df = BeWatch.db.get_video_df()

    # Load the cached sbvdf so we don't waste time resyncing
    sbvdf = BeWatch.db.get_synced_behavior_and_video_df()

    # Find the best overlap
    new_sbvdf = BeWatch.db.find_best_overlap_video(
        behavior_files_df, video_files_df,
        cached_sbvdf=sbvdf,
        always_prefer_mkv=True)
        
    # locale-ify
    new_sbvdf['filename'] = new_sbvdf['filename'].str.replace(
        PATHS['behavior_dir'], '$behavior_dir$')    
    new_sbvdf['filename_video'] = new_sbvdf['filename_video'].str.replace(
        PATHS['video_dir'], '$video_dir$')    
        
    # Save
    filename = os.path.join(PATHS['database_root'], 'behave_and_video.csv')
    new_sbvdf.to_csv(filename, index=False)

def daily_update_trial_matrix(start_date=None, verbose=False):
    """Cache the trial matrix for every session
    
    TODO: use cache
    """
    PATHS = BeWatch.db.get_paths()
    # Get
    behavior_files_df = BeWatch.db.get_behavior_df()
    
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
    PATHS = BeWatch.db.get_paths()
    # Get
    behavior_files_df = BeWatch.db.get_behavior_df()

    # Filter by those after start date
    behavior_files_df = behavior_files_df[ 
        behavior_files_df.dt_start >= start_date]

    # Load what we've already calculated
    pmdf = BeWatch.db.get_perf_metrics()

    # Calculate any that need it
    new_pmdf_rows_l = []
    for idx, brow in behavior_files_df.iterrows():
        # Check if it already exists
        session = brow['session']
        if session in pmdf['session'].values:
            if verbose:
                print "skipping", session
            continue
        
        # Skip anything that is not TwoChoice
        if brow['protocol'] != 'TwoChoice':
            continue
        
        # Otherwise run
        trial_matrix = BeWatch.db.get_trial_matrix(session)
        metrics = BeWatch.db.calculate_perf_metrics(trial_matrix)
        
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

