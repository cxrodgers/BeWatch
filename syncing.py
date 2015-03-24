"""Module for syncing behavioral and video files"""
import os
import numpy as np
import pandas
import ArduFSM
import my
from ArduFSM import TrialMatrix, TrialSpeak, mainloop
import BeWatch

def index_of_biggest_diffs_across_arr(ser, ncuts_total=3):
    """Return indices of biggest diffs in various segments of arr"""
    # Cut the series into equal length segments, not including NaNs
    ser = ser.dropna()
    cuts = [ser.index[len(ser) * ncut / ncuts_total] 
        for ncut in range(ncuts_total)]
    cuts.append(ser.index[-1])

    # Iterate over cuts and choose the index preceding the largest gap in the cut
    res = []
    for ncut in range(len(cuts) - 1):
        subser = ser.ix[cuts[ncut]:cuts[ncut+1]]
        res.append(subser.diff().shift(-1).argmax())
    return np.asarray(res)

def generate_test_times_for_user(times, max_time, initial_guess=(.9991, 7.5), 
    N=3, buffer=30):
    """Figure out the best times for a user to identify in the video
    
    times: Series of times in the initial time base.
    initial_guess: linear poly to apply to times as a first guess
    N: number of desired times, taken equally across video
    
    Returns the best times to check (those just before a large gap),
    in the guessed timebase.
    """
    # Apply the second guess, based on historical bias of above method
    new_values = np.polyval(initial_guess, times)
    times = pandas.Series(new_values, index=times.index)
    
    # Mask trials too close to end
    mask_by_buffer_from_end(times, max_time, buffer=buffer)

    # Identify the best trials to use for manual realignment
    test_idxs = index_of_biggest_diffs_across_arr(
        times, ncuts_total=N)
    test_times = times.ix[test_idxs]
    test_next_times = times.shift(-1).ix[test_idxs]
    
    return test_times, test_next_times
    

def mask_by_buffer_from_end(ser, end_time, buffer=10):
    """Set all values of ser to np.nan that occur within buffer of the ends"""
    ser[ser < buffer] = np.nan
    ser[ser > end_time - buffer] = np.nan

def generate_mplayer_guesses_and_sync(metadata, 
    user_results=None, guess=(1., 0.), N=4, pre_time=10):
    """Generates best times to check video, and potentially also syncs.
    
    metadata : a row from bv_files to sync
    
    The fit is between these datasets:
        X : time of retraction from behavior file, minus the test_guess_vvsb
            in the metadata.
        Y : user-supplied times of retraction from video
    The purpose of 'initial_guess' is to generate better guesses for the user
    to look in the video, but the returned data always use the combined fit
    that includes any initial guess. However, test_guess_vvsb is not
    accounted for in the returned value.
    
    N times to check in the video are printed out. Typically this is run twice,
    once before checking, then check, then run again now specifying the 
    video times in `user_results`.

    If the initial guess is very wrong, you may need to find a large
    gap in the video and match it up to trials info manually, and use this
    to fix `guess` to be closer.
    """
    initial_guess = np.asarray(guess)
    
    # Load trials info
    trials_info = TrialMatrix.make_trial_matrix_from_file(metadata['filename'])
    splines = TrialSpeak.load_splines_from_file(metadata['filename'])
    lines = TrialSpeak.read_lines_from_file(metadata['filename'])
    parsed_df_split_by_trial = \
        TrialSpeak.parse_lines_into_df_split_by_trial(lines)

    # Insert servo retract time
    trials_info['time_retract'] = TrialSpeak.identify_servo_retract_times(
        parsed_df_split_by_trial)

    # Apply the delta-time guess to the retraction times
    test_guess_vvsb = metadata['guess_vvsb_start'] #/ np.timedelta64(1, 's')
    trials_info['time_retract_vbase'] = \
        trials_info['time_retract'] - test_guess_vvsb

    # Choose test times for user
    video_duration = metadata['duration_video'] / np.timedelta64(1, 's')
    test_times, test_next_times = generate_test_times_for_user(
        trials_info['time_retract_vbase'], video_duration,
        initial_guess=initial_guess, N=N)

    # Print mplayer commands
    for test_time, test_next_time in zip(test_times, test_next_times):
        pre_test_time = int(test_time) - pre_time
        print 'mplayer -ss %d %s # guess %0.1f, next %0.1f' % (pre_test_time, 
            metadata['filename_video'], test_time, test_next_time)

    # If no data provided, just return
    if user_results is None:
        return {'test_times': test_times}
    if len(user_results) != N:
        print "warning: len(user_results) should be %d not %d" % (
            N, len(user_results))
        return {'test_times': test_times}
    
    # Otherwise, fit a correction to the original guess
    new_fit = np.polyfit(test_times.values, user_results, deg=1)
    resids = np.polyval(new_fit, test_times.values) - user_results

    # Composite the two fits
    # For some reason this is not transitive! This one appears correct.
    combined_fit = np.polyval(np.poly1d(new_fit), np.poly1d(initial_guess))

    # Diagnostics
    print os.path.split(metadata['filename'])[-1]
    print os.path.split(metadata['filename_video'])[-1]
    print "combined_fit: %r" % np.asarray(combined_fit)
    print "resids: %r" % np.asarray(resids)    
    
    return {'test_times': test_times, 'resids': resids, 
        'combined_fit': combined_fit}




## Begin house light syncing
def extract_onsets_and_durations(lums, delta=30, diffsize=3, refrac=5):
    """Extract house light times.
    
    First, we diff the sig to find onsets or offsets of at least delta
    within diffisze frames. Drop any that occur within refrac.
    
    Then we calculate the duration of each.
    
    Returns: onsets, durations
    """
    # diff the sig
    # maybe a better way is to boxcar by 30frames first
    diffsig = lums[diffsize:] - lums[:-diffsize]
    onsets = np.where(diffsig > delta)[0]
    offsets = np.where(diffsig < -delta)[0]
    
    # drop refractory onsets, offsets
    onsets2 = drop_refrac(onsets, refrac)
    offsets2 = drop_refrac(offsets, refrac)    
    
    # get durations
    remaining_onsets, durations = extract_duration_of_onsets(onsets2, offsets2)
    
    return remaining_onsets, durations
    

def drop_refrac(arr, refrac):
    """Drop all values in arr after a refrac from an earlier val"""
    drop_mask = np.zeros_like(arr).astype(np.bool)
    for idx, val in enumerate(arr):
        drop_mask[(arr < val + refrac) & (arr > val)] = 1
    return arr[~drop_mask]

def extract_duration_of_onsets(onsets, offsets):
    """Extract duration of each onset.
    
    This is the time to the next offset. If there is another intervening 
    onset, then drop the first one.
    
    Returns: remaining_onsets, durations
    """
    onsets3 = []
    durations = []
    for idx, val in enumerate(onsets):
        # Find upcoming offsets and skip if none
        upcoming_offsets = offsets[offsets > val]
        if len(upcoming_offsets) == 0:
            continue
        next_offset = upcoming_offsets[0]
        
        # Find upcoming onsets and skip if there is one before next offset
        upcoming_onsets = onsets[onsets > val]
        if len(upcoming_onsets) > 0 and upcoming_onsets[0] < next_offset:
            continue
        
        # Store duration and this onset
        onsets3.append(val)
        durations.append(next_offset - val)    

    return np.asarray(onsets3), np.asarray(durations)

def get_light_times_from_behavior_file(session):
    """Return time light goes on and off in logfile from session"""
    lines = BeWatch.db.get_logfile_lines(session)

    # They turn on in ERROR (14), INTER_TRIAL_INTERVAL (13), 
    # and off in ROTATE_STEPPER1 (2)
    parsed_df_by_trial = TrialSpeak.parse_lines_into_df_split_by_trial(lines)
    light_on = TrialSpeak.identify_state_change_times(
        parsed_df_by_trial, state1=[13, 14])
    light_off = TrialSpeak.identify_state_change_times(
        parsed_df_by_trial, state0=2)
    
    return light_on, light_off

def longest_unique_fit(xdata, ydata, start_fitlen=3, ss_thresh=.0003,
    verbose=True):
    """Find the longest consecutive string of fit points between x and y.

    # Finally, sync up
    # How about, start with N=5 trials from the middle of the behavior, 
    # find optimal matching 5 consecutive hits from house light
    # by scanning over all possible matches and keeping under some resid, 
    # increase N until only one unique hit foudn

    # If SS_THRESH is too generous, then we'll start including bad data points
    # at the ends and potentially corrupt the fit
    # Perhaps add another outer loop where we start with a very tight SS_THRESH
    # and increase if no fit found, or only fit of a very short length.
    # Also, could add an intermediate loop with various different starting
    # indexes, in case the middle of the session is corrupted.  

    Note ss_thresh is in terms of the units of y, eg frames or sec.
    
    Returns: best fit poly, or None if none found
    """
    # Choose the idx to start with in behavior
    fitlen = start_fitlen
    mid_idx = len(ydata) / 2
    keep_going = True
    best_fitpoly = None

    while keep_going:
        # Choose the data to fit
        chosen_idxs = xdata[mid_idx - fitlen:mid_idx + fitlen]
        
        # Check if we ran out of data
        if len(chosen_idxs) != fitlen * 2:
            break
        if np.any(np.isnan(chosen_idxs)):
            break
            

        # Find the best consecutive fit among onsets
        rec_l = []
        for idx in range(0, len(ydata) - len(chosen_idxs) + 1):
            # The data to fit with
            test = ydata[idx:idx + len(chosen_idxs)]
            if np.any(np.isnan(test)):
                continue
            
            # fit
            fitpoly = np.polyfit(test, chosen_idxs, deg=1)
            fit_to_input = np.polyval(fitpoly, test)
            resids = chosen_idxs - fit_to_input
            ss = np.sum(resids ** 2)
            rec_l.append({'idx': idx, 'ss': ss, 'fitpoly': fitpoly})
        
        # Look at results
        rdf = pandas.DataFrame.from_records(rec_l).set_index('idx').dropna()

        # Keep only those under thresh
        rdf = rdf[rdf['ss'] < ss_thresh * len(chosen_idxs)]    

        # If no fits, then quit
        if len(rdf) == 0:
            keep_going = False
            break
        
        # Take the best fit
        best_index = rdf['ss'].argmin()
        best_ss = rdf['ss'].min()
        best_fitpoly = rdf['fitpoly'].ix[best_index]

        # Increase the size
        fitlen = fitlen + 1    
    
    if verbose:
        print "fit found,", fitlen
    
    return best_fitpoly



def get_or_save_lums(session, lumdir=None):
    """Load lum for session from video or if available from cache"""    
    PATHS = BeWatch.db.get_paths()
    if lumdir is None:
        lumdir = os.path.join(PATHS['database_root'], 'lums')
    
    # Get metadata about session
    sbvdf = BeWatch.db.get_synced_behavior_and_video_df().set_index('session')
    session_row = sbvdf.ix[session]
    guess_vvsb_start = session_row['guess_vvsb_start']
    vfilename = session_row['filename_video']
    
    # New style filenames
    new_lum_filename = os.path.join(lumdir, 
        os.path.split(vfilename)[1] + '.lums')
    
    # If new exists, return
    if os.path.exists(new_lum_filename):
        print "cached lums found"
        lums = my.misc.pickle_load(new_lum_filename)
        return lums    

    # Get the lums ... this takes a while
    lums = my.misc.process_chunks_of_video(vfilename, n_frames=np.inf)
    
    # Save
    my.misc.pickle_dump(lums, new_lum_filename)
    
    return lums
    

def autosync_behavior_and_video_with_houselight(session, save_result=True,
    light_delta=30):
    """Main autosync function
    
    Loads lums and behavioral onsets from session.
    Runs through the longest_unique_fit function with various ss_thresh
    Stores if fit found
    
    save_result: whether to write to db
    light_delta: size of lum delta to detect
        30 works well for dim lamp; 4 if direct IR illum
    """
    # Get metadata about session
    sbvdf = BeWatch.db.get_synced_behavior_and_video_df().set_index('session')
    session_row = sbvdf.ix[session]
    guess_vvsb_start = session_row['guess_vvsb_start']
    vfilename = session_row['filename_video']

    # Get the lums ... this takes a while
    lums = get_or_save_lums(session)

    # Get onsets and durations
    onsets, durations = extract_onsets_and_durations(lums, 
        delta=light_delta, diffsize=3, refrac=5)

    # Same data from ardulines
    light_on, light_off = get_light_times_from_behavior_file(session)

    # Subtract test_guess_vvsb from the behavior data
    light_on = light_on - guess_vvsb_start
    light_off = light_off - guess_vvsb_start

    # Divide by XXX fps in the video data
    onsets = onsets / 29.97
    durations = durations / 29.97
    
    # TODO: Try subtracting the video duration from the video timestamps,
    # I bet it will result in a near zero offset.
    
    # A crude initial guess strategy
    # We need to get rid of the ones that were clearly from other sessions
    # in the same video. We've already applied vvsb guess to light_on,
    # the behavioral data, so use the min and max from this to mask the 
    # video data
    vmask_min = np.nanmin(light_on)
    vmask_max = np.nanmax(light_on)
    masked_onsets = onsets[(onsets > vmask_min) & (onsets < vmask_max)]

    # Fit from behavior to video, because we want to start with good
    # video onsets and go from there.
    for ss_thresh in [.0003, .001, .003]:
        best_fitpoly = longest_unique_fit(light_on, masked_onsets, 
            start_fitlen=3, ss_thresh=ss_thresh)
        if best_fitpoly is not None:
            break

    # Invert fit to go from video to behavior
    if best_fitpoly is None:
        print "warning: cannot sync", session
        return None
    fit_v2b = my.misc.invert_linear_poly(best_fitpoly)

    # Store
    if save_result:
        if fit_v2b is not None:
            BeWatch.db.set_manual_bv_sync(session, fit_v2b)

    return fit_v2b


def autosync_behavior_and_video_with_houselight_from_day(date=None):
    """Autosync all sessions using house light from specified date"""
    # Load metadata
    msdf = BeWatch.db.get_manual_sync_df()
    sbvdf = BeWatch.db.get_synced_behavior_and_video_df()
    sbvdf_dates = sbvdf['dt_end'].apply(lambda dt: dt.date())
    
    # Set to most recent date in database if None
    if date is None:
        date = sbvdf_dates.max()
    
    # Choose the ones to display
    display_dates = sbvdf.ix[sbvdf_dates == date]
    if len(display_dates) > 20:
        raise ValueError("too many dates")

    for idx, row in display_dates.iterrows():
        session = row['session']
        if session in msdf.index:
            print session, "already synced"
        else:
            print session
            autosync_behavior_and_video_with_houselight(session, 
                save_result=True)
