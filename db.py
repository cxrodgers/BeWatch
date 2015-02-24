"""Maintaining the database of behavioral data

"""
import os
import numpy as np
import glob
import re
import pandas
import subprocess # for ffprobe
import ArduFSM
import scipy.misc
import my
import datetime
from ArduFSM import TrialMatrix, TrialSpeak, mainloop
import socket


def get_locale():
    """Return the hostname"""
    return socket.gethostname()

def get_paths():
    """Return the data directories on this locale"""
    LOCALE = get_locale()
    if LOCALE == 'chris-pyramid':
        PATHS = {
            'database_root': '/home/chris/mnt/marvin/dev/behavior_db',
            'behavior_dir': '/home/chris/mnt/marvin/runmice',
            'video_dir': '/home/chris/mnt/marvin/compressed_eye',
            }

    elif LOCALE == 'marvin':
        PATHS = {
            'database_root': '/home/mouse/dev/behavior_db',
            'behavior_dir': '/home/mouse/runmice',
            'video_dir': '/home/mouse/compressed_eye',
            }

    else:
        raise ValueError("unknown locale %s" % LOCALE)
    
    return PATHS

def getstarted():
    """Return a dict of data about locale, paths, and mice."""
    res = {
        'locale': get_locale(),
        'paths': get_paths(),
        }
    
    res['mice'] = ['AM03', 'AM05', 'KF13', 'KM14', 'KF16', 'KF17', 'KF18', 'KF19', 
        'KM24', 'KM25', 'KF26', 'KF28', 'KF30', 'KF32', 'KF33', 'KF35', 'KF36',
        'KF37']
    
    res['rigs'] = ['L1', 'L2', 'L3']
    
    res['aliases'] = {
        'KF13A': 'KF13',
        'AM03A': 'AM03',
        }

    # Known mice
    assert np.all([alias_val in res['mice'] 
        for alias_val in res['aliases'].values()])
    
    return res

def check_ardulines(logfile):
    """Error check the log file.
    
    Here are the things that would be useful to check:
    * File can be loaded with the loading function without error.
    * Reported lick times match reported choice
    * Reported choice and stimulus matches reported outcome
    * All lines are (time, arg, XXX) where arg is known
    * All state transitions are legal, though this is a lot of work
    
    Diagnostics to return
    * Values of params over trials
    * Various stats on time spent in each state
    * Empirical state transition probabilities
    
    Descriptive metrics of each trial
    * Lick times locked to response window opening
    * State transition times locked to response window opening
    """
    # Make sure the loading functions work
    lines = TrialSpeak.read_lines_from_file(logfile)
    pldf = TrialSpeak.parse_lines_into_df(lines)
    plst = TrialSpeak.parse_lines_into_df_split_by_trial(lines)

def get_perf_metrics():
    """Return the df of perf metrics over sessions"""
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'perf_metrics.csv')

    try:
        pmdf = pandas.read_csv(filename)
    except IOError:
        raise IOError("cannot find perf metrics database at %s" % filename)
    
    return pmdf

def flush_perf_metrics():
    """Create an empty perf metrics file"""
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'perf_metrics.csv')
    columns=['session', 'n_trials', 'spoil_frac',
        'perf_all', 'perf_unforced',
        'fev_corr_all', 'fev_corr_unforced',
        'fev_side_all', 'fev_side_unforced',
        'fev_stay_all','fev_stay_unforced',
        ]

    pmdf = pandas.DataFrame(np.zeros((0, len(columns))), columns=columns)
    pmdf.to_csv(filename, index=False)

def get_logfile_lines(session):
    """Look up the logfile for a session and return it"""
    # Find the filename
    bdf = get_behavior_df()
    rows = bdf[bdf.session == session]
    if len(rows) != 1:
        raise ValueError("cannot find unique session for %s" % session)
    filename = rows.irow(0)['filename']
    
    # Read lines
    lines = TrialSpeak.read_lines_from_file(filename)
    
    # Split by trial
    #~ splines = split_by_trial(lines)
    
    return lines

def get_trial_matrix(session):
    """Return the (cached) trial matrix for a session"""
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'trial_matrix', session)
    res = pandas.read_csv(filename)
    return res

def get_all_trial_matrix():
    """Return a dict of all cached trial matrices"""
    PATHS = get_paths()
    all_filenames = glob.glob(os.path.join(
        PATHS['database_root'], 'trial_matrix', '*'))
    
    session2trial_matrix = {}
    for filename in all_filenames:
        session = os.path.split(filename)[1]
        trial_matrix = pandas.read_csv(filename)
        session2trial_matrix[session] = trial_matrix
    
    return session2trial_matrix

def get_behavior_df():
    """Returns the current behavior database"""
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'behavior.csv')

    try:
        behavior_files_df = pandas.read_csv(filename, 
            parse_dates=['dt_end', 'dt_start', 'duration'])
    except IOError:
        raise IOError("cannot find behavior database at %s" % filename)
    
    # de-localeify
    behavior_files_df['filename'] = behavior_files_df['filename'].str.replace(
        '\$behavior_dir\$', PATHS['behavior_dir'])
    
    # Alternatively, could store as floating point seconds
    behavior_files_df['duration'] = pandas.to_timedelta(
        behavior_files_df['duration'])
    
    return behavior_files_df
    
def get_video_df():
    """Returns the current video database"""
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'video.csv')

    try:
        video_files_df = pandas.read_csv(filename,
            parse_dates=['dt_end', 'dt_start'])
    except IOError:
        raise IOError("cannot find video database at %s" % filename)

    # de-localeify
    video_files_df['filename'] = video_files_df['filename'].str.replace(
        '\$video_dir\$', PATHS['video_dir'])
    
    # Alternatively, could store as floating point seconds
    video_files_df['duration'] = pandas.to_timedelta(
        video_files_df['duration'])    
    
    return video_files_df

def get_synced_behavior_and_video_df():
    """Return the synced behavior/video database"""
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'behave_and_video.csv')
    
    try:
        synced_bv_df = pandas.read_csv(filename, parse_dates=[
            'dt_end', 'dt_start', 'dt_end_video', 'dt_start_video'])
    except IOError:
        raise IOError("cannot find synced database at %s" % filename)
    
    # Alternatively, could store as floating point seconds
    synced_bv_df['duration'] = pandas.to_timedelta(
        synced_bv_df['duration'])    
    synced_bv_df['duration_video'] = pandas.to_timedelta(
        synced_bv_df['duration_video'])    

    # de-localeify
    synced_bv_df['filename_video'] = synced_bv_df['filename_video'].str.replace(
        '\$video_dir\$', PATHS['video_dir'])
    synced_bv_df['filename'] = synced_bv_df['filename'].str.replace(
        '\$behavior_dir\$', PATHS['behavior_dir'])        
    
    return synced_bv_df    

def get_manual_sync_df():
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'manual_bv_sync.csv')
    
    try:
        manual_bv_sync = pandas.read_csv(filename).set_index('session')
    except IOError:
        raise IOError("cannot find manual sync database at %s" % filename)    
    
    return manual_bv_sync

def set_manual_bv_sync(session, sync_poly):
    """Store the manual behavior-video sync for session
    
    TODO: also store guess_vvsb, even though it's redundant with
    the main sync df. These fits are relative to that.
    """
    PATHS = get_paths()
    
    # Load any existing manual results
    manual_sync_df = get_manual_sync_df()
    
    sync_poly = np.asarray(sync_poly) # indexing is backwards for poly
    
    # Add
    if session in manual_sync_df.index:
        raise ValueError("sync already exists for %s" % session)
    
    manual_sync_df = manual_sync_df.append(
        pandas.DataFrame([[sync_poly[0], sync_poly[1]]],
            index=[session],
            columns=['fit0', 'fit1']))
    manual_sync_df.index.name = 'session' # it forgets
    
    # Store
    filename = os.path.join(PATHS['database_root'], 'manual_bv_sync.csv')
    manual_sync_df.to_csv(filename)

def interactive_bv_sync():
    """Interactively sync behavior and video"""
    # Load synced data
    sbvdf = get_synced_behavior_and_video_df()
    msdf = get_manual_sync_df()
    sbvdf = sbvdf.join(msdf, on='session')

    # Choose session
    choices = sbvdf[['session', 'dt_start', 'best_video_overlap', 'rig', 'fit1']]
    choices = choices.rename(columns={'best_video_overlap': 'vid_overlap'})

    print "Here are the most recent sessions:"
    print choices[-20:]
    choice = None
    while choice is None:
        choice = raw_input('Which index to analyze? ')
        try:
            choice = int(choice)
        except ValueError:
            pass
    test_row = sbvdf.ix[choice]

    # Run sync
    N_pts = 3
    sync_res0 = generate_mplayer_guesses_and_sync(test_row, N=N_pts)

    # Get results
    n_results = []
    for n in range(N_pts):
        res = raw_input('Enter result: ')
        n_results.append(float(res))

    # Run sync again
    sync_res1 = generate_mplayer_guesses_and_sync(test_row, N=N_pts,
        user_results=n_results)

    # Store
    res = raw_input('Confirm insertion [y/N]? ')
    if res == 'y':
        set_manual_bv_sync(test_row['session'], 
            sync_res1['combined_fit'])
        print "inserted"
    else:
        print "not inserting"    



## End of database stuff




def calculate_pivoted_performances(start_date=None, delta_days=15):
    """Returns pivoted performance metrics"""
    # Choose start date
    if start_date is None:
        start_date = datetime.datetime.now() - \
            datetime.timedelta(days=delta_days)    
    
    # Get data and add the mouse column
    bdf = my.behavior.get_behavior_df()
    pmdf = my.behavior.get_perf_metrics()
    pmdf = pmdf.join(bdf.set_index('session')[['mouse', 'dt_start']], on='session')
    pmdf = pmdf.ix[pmdf.dt_start >= start_date].drop('dt_start', 1)
    #pmdf.index = range(len(pmdf))

    # always sort on session
    pmdf = pmdf.sort('session')

    # add a "date_s" column which is just taken from the session for now
    pmdf['date_s'] = pmdf['session'].str[2:8]
    pmdf['date_s'] = pmdf['date_s'].apply(lambda s: s[:2]+'-'+s[2:4]+'-'+s[4:6])

    # Check for duplicate sessions for a given mouse
    # This gives you the indices to drop, so confusingly, take_last means
    # it returns idx of the first
    # We want to keep the last of the day (??) so take_first
    dup_idxs = pmdf[['date_s', 'mouse']].duplicated(take_last=False)
    if dup_idxs.sum() > 0:
        print "warning: dropping %d duplicated sessions" % dup_idxs.sum()
        print "\n".join(pmdf['session'][dup_idxs].values)
        pmdf = pmdf.drop(pmdf.index[dup_idxs])

    # pivot on all metrics
    piv = pmdf.drop('session', 1).pivot_table(index='mouse', columns='date_s')

    # Find missing data
    missing_data = piv['n_trials'].isnull().unstack()
    missing_data = missing_data.ix[missing_data].reset_index()
    missing_rows = []
    for idx, row in missing_data.iterrows():
        missing_rows.append(row['date_s'] + ' ' + row['mouse'])
    if len(missing_rows) > 0:
        print "warning: missing the following sessions:"
        print "\n".join(missing_rows)
    
    return piv

def calculate_pivoted_perf_by_rig(start_date=None, delta_days=15, 
    drop_mice=None):
    """Pivot performance by rig and day"""
    # Choose start date
    if start_date is None:
        start_date = datetime.datetime.now() - \
            datetime.timedelta(days=delta_days)    
    
    # Get behavior data
    bdf = my.behavior.get_behavior_df()
    
    # Get perf columns of interest and join on rig and date
    pmdf = my.behavior.get_perf_metrics()[[
        'session', 'perf_unforced', 'n_trials', 'fev_side_unforced']]
    pmdf = pmdf.join(bdf.set_index('session')[['rig', 'dt_start', 'mouse']], 
        on='session')
    pmdf = pmdf.ix[pmdf.dt_start >= start_date].drop('dt_start', 1)

    # always sort on session
    pmdf = pmdf.sort('session')

    # add a "date_s" column which is just taken from the session for now
    pmdf['date_s'] = pmdf['session'].str[2:8]
    pmdf['date_s'] = pmdf['date_s'].apply(lambda s: s[:2]+'-'+s[2:4]+'-'+s[4:6])

    # drop by mice
    if drop_mice is not None:
        pmdf = pmdf[~pmdf['mouse'].isin(drop_mice)]
    pmdf = pmdf.drop('mouse', 1)

    # pivot on all metrics, and mean over replicates
    piv = pmdf.drop('session', 1).pivot_table(index='rig', columns='date_s')
    
    return piv

def calculate_perf_metrics(trial_matrix):
    """Calculate simple performance metrics on a session"""
    rec = {}
    
    # Trials and spoiled fraction
    rec['n_trials'] = len(trial_matrix)
    rec['spoil_frac'] = float(np.sum(trial_matrix.outcome == 'spoil')) / \
        len(trial_matrix)

    # Calculate performance
    rec['perf_all'] = float(len(my.pick(trial_matrix, outcome='hit'))) / \
        len(my.pick(trial_matrix, outcome=['hit', 'error']))
    
    # Calculate unforced performance, protecting against low trial count
    n_nonbad_nonspoiled_trials = len(
        my.pick(trial_matrix, outcome=['hit', 'error'], isrnd=True))
    if n_nonbad_nonspoiled_trials < 10:
        rec['perf_unforced'] = np.nan
    else:
        rec['perf_unforced'] = float(
            len(my.pick(trial_matrix, outcome='hit', isrnd=True))) / \
            n_nonbad_nonspoiled_trials

    # Anova with and without remove bad
    for remove_bad in [True, False]:
        # Numericate and optionally remove non-random trials
        numericated_trial_matrix = TrialMatrix.numericate_trial_matrix(
            trial_matrix)
        if remove_bad:
            suffix = '_unforced'
            numericated_trial_matrix = numericated_trial_matrix.ix[
                numericated_trial_matrix.isrnd == True]
        else:
            suffix = '_all'
        
        # Run anova
        aov_res = TrialMatrix._run_anova(numericated_trial_matrix)
        
        # Parse FEV
        if aov_res is not None:
            rec['fev_stay' + suffix], rec['fev_side' + suffix], \
                rec['fev_corr' + suffix] = aov_res['ess'][
                ['ess_prevchoice', 'ess_Intercept', 'ess_rewside']]
        else:
            rec['fev_stay' + suffix], rec['fev_side' + suffix], \
                rec['fev_corr' + suffix] = np.nan, np.nan, np.nan    
    
    return rec





def search_for_behavior_files(behavior_dir='~/mnt/behave/runmice',
    clean=True):
    """Load behavior files into data frame.
    
    behavior_dir : where to look
    clean : see parse_behavior_filenames
    
    See also search_for_behavior_and_video_files
    """
    # expand path
    behavior_dir = os.path.expanduser(behavior_dir)
    
    # Acquire all behavior files in the subdirectories
    all_behavior_files = []
    for subdir in rigs:
        all_behavior_files += glob.glob(os.path.join(
            behavior_dir, subdir, 'logfiles', 'ardulines.*'))

    # Parse out metadata for each
    behavior_files_df = parse_behavior_filenames(all_behavior_files, 
        clean=clean)    
    
    # Sort and reindex
    behavior_files_df = behavior_files_df.sort('dt_start')
    behavior_files_df.index = range(len(behavior_files_df))
    
    return behavior_files_df

def search_for_behavior_and_video_files(
    behavior_dir='~/mnt/behave/runmice',
    video_dir='~/mnt/bruno-nix/compressed_eye',
    cached_video_files_df=None,
    ):
    """Get a list of behavior and video files, with metadata.
    
    Looks for all behavior directories in behavior_dir/rignumber.
    Looks for all video files in video_dir (using cache).
    Gets metadata about video files using parse_video_filenames.
    Finds which video file maximally overlaps with which behavior file.
    
    Returns: joined, video_files_df
        joined is a data frame with the following columns:
            u'dir', u'dt_end', u'dt_start', u'duration', u'filename', 
            u'mouse', u'rig', u'best_video_index', u'best_video_overlap', 
            u'dt_end_video', u'dt_start_video', u'duration_video', 
            u'filename_video', u'rig_video'
        video_files_df is basically used only to re-cache
    """
    # expand path
    behavior_dir = os.path.expanduser(behavior_dir)
    video_dir = os.path.expanduser(video_dir)

    # Search for behavior files
    behavior_files_df = search_for_behavior_files(behavior_dir)

    # Acquire all video files
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    if len(video_files) == 0:
        print "warning: no video files found"
    video_files_df = parse_video_filenames(video_files, verbose=True,
        cached_video_files_df=cached_video_files_df)

    # Find the best overlap
    new_behavior_files_df = find_best_overlap_video(
        behavior_files_df, video_files_df)
    
    # Join video info
    joined = new_behavior_files_df.join(video_files_df, 
        on='best_video_index', rsuffix='_video')    
    
    return joined, video_files_df

def find_best_overlap_video(behavior_files_df, video_files_df):
    """Find the video file with the best overlap for each behavior file.
    
    Returns : behavior_files_df, but now with a best_video_index and
        a best_video_overlap columns. Suitable for the following:
        behavior_files_df.join(video_files_df, on='best_video_index', 
            rsuffix='_video')
    """
    # Operate on a copy
    behavior_files_df = behavior_files_df.copy()
    
    # Find behavior files that overlapped with video files
    behavior_files_df['best_video_index'] = -1
    behavior_files_df['best_video_overlap'] = 0.0
    
    # Something is really slow in this loop
    for bidx, brow in behavior_files_df.iterrows():
        # Find the overlap between this behavioral session and video sessions
        # from the same rig
        latest_start = video_files_df[
            video_files_df.rig == brow['rig']]['dt_start'].copy()
        latest_start[latest_start < brow['dt_start']] = brow['dt_start']
            
        earliest_end = video_files_df[
            video_files_df.rig == brow['rig']]['dt_end'].copy()
        earliest_end[earliest_end > brow['dt_end']] = brow['dt_end']
        
        # Find the video with the most overlap
        overlap = (earliest_end - latest_start)
        if len(overlap) == 0:
            # ie, no video files found
            continue
        vidx_max_overlap = overlap.argmax()
        
        # Convert from numpy timedelta64 to a normal number
        max_overlap_sec = overlap.ix[vidx_max_overlap] / np.timedelta64(1, 's')
        
        # Store if it's more than zero
        if max_overlap_sec > 0:
            behavior_files_df.loc[bidx, 'best_video_index'] = vidx_max_overlap
            behavior_files_df.loc[bidx, 'best_video_overlap'] = max_overlap_sec

    return behavior_files_df

def parse_behavior_filenames(all_behavior_files, clean=True):
    """Given list of ardulines files, extract metadata and return as df.
    
    Each filename is matched to a pattern which is used to extract the
    rigname, date, and mouse name. Non-matching filenames are discarded.
    
    clean : if True, also clean up the mousenames by upcasing and applying
        aliases. Finally, drop the ones not in the official list of mice.
    """
    # Extract info from filename
    # directory, rigname, datestring, mouse
    pattern = '(\S+)/(\S+)/logfiles/ardulines\.(\d+)\.(\S+)'
    rec_l = []
    for filename in all_behavior_files:
        # Match filename pattern
        m = re.match(pattern, os.path.abspath(filename))
        if m is not None:
            dir, rig, date_s, mouse = m.groups()

            # The start time is parsed from the filename
            date = datetime.datetime.strptime(date_s, '%Y%m%d%H%M%S')
            
            # The end time is parsed from the file timestamp
            behavior_end_time = datetime.datetime.fromtimestamp(
                my.misc.get_file_time(filename))
            
            # Store
            rec_l.append({'rig': rig, 'mouse': mouse,
                'dt_start': date, 'dt_end': behavior_end_time,
                'duration': behavior_end_time - date,
                'filename': filename})
    behavior_files_df = pandas.DataFrame.from_records(rec_l)

    if len(behavior_files_df) == 0:
        print "warning: no behavior files found"

    elif clean:
        # Clean the behavior files by upcasing and applying aliases
        behavior_files_df.mouse = behavior_files_df.mouse.apply(str.upper)
        behavior_files_df.mouse.replace(aliases, inplace=True)

        # Drop any that are not in the list of accepted mouse names
        behavior_files_df = behavior_files_df.ix[behavior_files_df.mouse.isin(mice)]

    # Add a session name based on the date and cleaned mouse name
    behavior_files_df['session'] = behavior_files_df['filename'].apply(
        lambda s: os.path.split(s)[1].split('.')[1]) + \
        '.' + behavior_files_df['mouse']

    return behavior_files_df

def parse_video_filenames(video_filenames, verbose=False, 
    cached_video_files_df=None):
    """Given list of video files, extract metadata and return df.

    For each filename, we extract the date (from the filename) and duration
    (using ffprobe).
    
    If cached_video_files_df is given:
        1) Checks that everything in cached_video_files_df.filename is also in
        video_filenames, else errors (because probably something
        has gone wrong, like the filenames are misformatted).
        2) Skips the probing of any video file already present in 
        cached_video_files_df
        3) Concatenates the new video files info with cached_video_files_df
        and returns.
    
    Returns:
        video_files_df, a DataFrame with the following columns: 
            dt_end dt_start duration filename rig
    """
    # Error check
    if cached_video_files_df is not None and not np.all([f in video_filenames 
        for f in cached_video_files_df.filename]):
        raise ValueError("cached_video_files contains unneeded video files")
    
    # Extract info from filename
    # directory, rigname, datestring, extension
    pattern = '(\S+)/(\S+)\.(\d+)\.(\S+)'
    rec_l = []

    for video_filename in video_filenames:
        if cached_video_files_df is not None and \
            video_filename in cached_video_files_df.filename.values:
            continue
        
        if verbose:
            print video_filename
        
        # Match filename pattern
        m = re.match(pattern, os.path.abspath(video_filename))
        if m is None:
            continue
        dir, rig, date_s, video_ext = m.groups()
        
        # Parse the end time using the datestring
        video_end_time = datetime.datetime.strptime(date_s, '%Y%m%d%H%M%S')

        # Video duration and hence start time
        proc = subprocess.Popen(['ffprobe', video_filename],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        res = proc.communicate()[0]

        # Check if ffprobe failed, probably on a bad file
        if 'Invalid data found when processing input' in res:
            # Just store what we know so far and warn
            rec_l.append({'filename': video_filename, 'rig': rig,
                'dt_end': video_end_time,
                })            
            if verbose:
                print "Invalid data found by ffprobe in %s" % video_filename
            continue

        # Parse out start time
        duration_match = re.search("Duration: (\S+),", res)
        assert duration_match is not None and len(duration_match.groups()) == 1
        video_duration_temp = datetime.datetime.strptime(
            duration_match.groups()[0], '%H:%M:%S.%f')
        video_duration = datetime.timedelta(
            hours=video_duration_temp.hour, 
            minutes=video_duration_temp.minute, 
            seconds=video_duration_temp.second,
            microseconds=video_duration_temp.microsecond)
        video_start_time = video_end_time - video_duration
        
        # Store
        rec_l.append({'filename': video_filename, 'rig': rig,
            'dt_end': video_end_time,
            'duration': video_duration,
            'dt_start': video_start_time,
            })

    resdf = pandas.DataFrame.from_records(rec_l)
    
    # Join with cache, if necessary
    if cached_video_files_df is not None:
        if len(resdf) == 0:
            resdf = cached_video_files_df
        else:
            resdf = pandas.concat([resdf, cached_video_files_df], axis=0, 
                ignore_index=True, verify_integrity=True)
    
    
    # Sort and reindex
    resdf = resdf.sort('dt_start')
    resdf.index = range(len(resdf))    
    
    return resdf

