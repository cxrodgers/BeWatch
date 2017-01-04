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
    # Avoiding setting with copy warning since we don't care
    ser.iscopy = False
    
    ser[ser < buffer] = np.nan
    ser[ser > end_time - buffer] = np.nan

def generate_mplayer_guesses_and_sync(metadata, 
    user_results=None, guess=(1., 0.), N=4, pre_time=10):
    """Generates best times to check video, and potentially also syncs.
    
    metadata : a row from bv_files to sync. Needs to specify the following:
        'filename' : behavioral filename
        'guess_vvsb_start'
        'duration_video'
        'filename_video'
    
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
def extract_onsets_and_durations(lums, delta=30, diffsize=3, refrac=5,
    verbose=False):
    """Identify sudden, sustained increments in the signal `lums`.
    
    Algorithm
    1.  Take the diff of lums over a period of `diffsize`.
        In code, this is: lums[diffsize:] - lums[:-diffsize]
        Note that this means we cannot detect an onset before `diffsize`.
        Also note that this "smears" sudden onsets, so later we will always
        take the earliest point.
    2.  Threshold this signal to identify onsets (indexes above 
        threshold) and offsets (indexes below -threshold). Add `diffsize`
        to each onset and offset to account for the shift incurred in step 1.
    3.  Drop consecutive onsets that occur within `refrac` samples of
        each other. Repeat separately for offsets. This is done with
        the function `drop_refrac`. Because this is done separately, note
        that if two increments are separated by a brief return to baseline,
        the second increment will be completely ignored (not combined with
        the first one).
    4.  Convert the onsets and offsets into onsets and durations. This is
        done with the function `extract duration of onsets2`. This discards
        any onset without a matching offset.
    
    TODO: consider applying a boxcar of XXX frames first.
    
    Returns: onsets, durations
        onsets : array of the onset of each increment, in samples.
            This will be the first sample that includes the detectable
            increment, not the sample before it.
        durations : array of the duration of each increment, in samples
            Same length as onsets. This is "Pythonic", so if samples 10-12
            are elevated but 9 and 13 are not, the onset is 10 and the duration
            is 3.
    """
    # diff the sig over a period of diffsize
    diffsig = lums[diffsize:] - lums[:-diffsize]
    
    # Threshold and account for the shift
    onsets = np.where(diffsig > delta)[0] + diffsize
    offsets = np.where(diffsig < -delta)[0] + diffsize
    if verbose:
        print onsets
    
    # drop refractory onsets, offsets
    onsets2 = drop_refrac(onsets, refrac)
    offsets2 = drop_refrac(offsets, refrac)    
    if verbose:
        print onsets2
    
    # get durations
    remaining_onsets, durations = extract_duration_of_onsets2(onsets2, offsets2)
    if verbose:
        print remaining_onsets
    
    return remaining_onsets, durations
    

def drop_refrac(arr, refrac):
    """Drop all values in arr after a refrac from an earlier val"""
    drop_mask = np.zeros_like(arr).astype(np.bool)
    for idx, val in enumerate(arr):
        drop_mask[(arr < val + refrac) & (arr > val)] = 1
    return arr[~drop_mask]

def extract_duration_of_onsets(onsets, offsets):
    """Extract duration of each onset.
    
    The duration is the time to the next offset. If there is another 
    intervening onset, then drop the first one.
    
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

def extract_duration_of_onsets2(onsets, offsets):
    """Extract duration of each onset.
    
    Use a "greedy" algorithm. For each onset:
        * Assign it to the next offset
        * Drop any intervening onsets
        * Continue with the next onset

    Returns: remaining_onsets, durations
    """
    onsets3 = []
    durations = []
    
    if len(onsets) == 0:
        return np.array([], dtype=np.int), np.array([], dtype=np.int)
    
    # This trigger will be set after each detected duration to mask out
    # subsequent onsets greedily
    onset_trigger = np.min(onsets) - 1
    
    # Iterate over onsets
    for idx, val in enumerate(onsets):
        # Skip onsets 
        if val < onset_trigger:
            continue
        
        # Find upcoming offsets and skip if none
        upcoming_offsets = offsets[offsets > val]
        if len(upcoming_offsets) == 0:
            continue
        next_offset = upcoming_offsets[0]
        
        # Store duration and this onset
        onsets3.append(val)
        durations.append(next_offset - val)
        
        # Save this trigger to skip subsequent onsets greedily
        onset_trigger = next_offset

    return np.asarray(onsets3), np.asarray(durations)


def get_light_times_from_behavior_file(session=None, logfile=None):
    """Return time light goes on and off in logfile from session"""
    if session is not None:
        lines = BeWatch.db.get_logfile_lines(session)
    elif logfile is not None:
        lines = TrialSpeak.read_lines_from_file(logfile)
    else:
        raise ValueError("must provide either session or logfile")

    # They turn on in ERROR (14), INTER_TRIAL_INTERVAL (13), 
    # and off in ROTATE_STEPPER1 (2)
    parsed_df_by_trial = TrialSpeak.parse_lines_into_df_split_by_trial(lines)
    light_on = TrialSpeak.identify_state_change_times(
        parsed_df_by_trial, state1=[13, 14], show_warnings=False)
    light_off = TrialSpeak.identify_state_change_times(
        parsed_df_by_trial, state0=2)
    
    return light_on, light_off

def longest_unique_fit(xdata, ydata, start_fitlen=3, ss_thresh=.0003,
    verbose=True, x_midslice_start=None):
    """Find the longest consecutive string of fit points between x and y.

    We start by taking a slice from xdata of length `start_fitlen` 
    points. This slice is centered at `x_midslice_start` (by default,
    halfway through). We then take all possible contiguous slices of 
    the same length from `ydata`; fit each one to the slice from `xdata`;
    and calculate the best-fit sum-squared residual per data point. 
    
    (Technically seems we are fitting from Y to X.)
    
    If any slices have a per-point residual less than `ss_thresh`, then 
    increment the length of the fit and repeat. If none do, then return 
    the best fit for the previous iteration, or None if this is the first
    iteration.
    
    Usually it's best to begin with a small ss_thresh, because otherwise
    bad data points can get incorporated at the ends and progressively worsen
    the fit. If no fit can be found, try increasing ss_thresh, or specifying a
    different x_midslice_start. Note that it will break if the slice in
    xdata does not occur anywhere in ydata, so make sure that the midpoint
    of xdata is likely to be somewhere in ydata.

    xdata, ydata : unmatched data to be fit
    start_fitlen : length of the initial slice
    ss_thresh : threshold sum-squared residual per data point to count
        as an acceptable fit. These will be in the units of X.
    verbose : issue status messages
    x_midslice_start : the center of the data to take from `xdata`. 
        By default, this is the midpoint of `xdata`.

    Returns: a linear polynomial fitting from Y to X.
    """
    # Choose the idx to start with in behavior
    fitlen = start_fitlen
    if x_midslice_start is None:
        x_midslice_start = len(xdata) / 2
    keep_going = True
    best_fitpoly = None

    if verbose:
        print "begin with fitlen", fitlen

    while keep_going:        
        # Slice out xdata
        chosen_idxs = xdata[x_midslice_start - fitlen:x_midslice_start + fitlen]
        
        # Check if we ran out of data
        if len(chosen_idxs) != fitlen * 2:
            if verbose:
                print "out of data, breaking"
            break
        if np.any(np.isnan(chosen_idxs)):
            if verbose:
                print "nan data, breaking"
            break

        # Find the best consecutive fit among onsets
        rec_l = []
        for idx in range(0, len(ydata) - len(chosen_idxs) + 1):
            # The data to fit with
            test = ydata[idx:idx + len(chosen_idxs)]
            if np.any(np.isnan(test)):
                # This happens when the last data point in ydata is nan
                continue
            
            # fit
            fitpoly = np.polyfit(test, chosen_idxs, deg=1)
            fit_to_input = np.polyval(fitpoly, test)
            resids = chosen_idxs - fit_to_input
            ss = np.sum(resids ** 2)
            rec_l.append({'idx': idx, 'ss': ss, 'fitpoly': fitpoly})
        
        # Test if there were no fits to analyze
        if len(rec_l) == 0:
            keep_going = False
            if verbose:
                print "no fits to threshold, breaking"
            break
        
        # Look at results
        rdf = pandas.DataFrame.from_records(rec_l).set_index('idx').dropna()

        # Keep only those under thresh
        rdf = rdf[rdf['ss'] < ss_thresh * len(chosen_idxs)]    

        # If no fits, then quit
        if len(rdf) == 0:
            keep_going = False
            if verbose:
                print "no fits under threshold, breaking"
            break
        
        # Take the best fit
        best_index = rdf['ss'].argmin()
        best_ss = rdf['ss'].min()
        best_fitpoly = rdf['fitpoly'].ix[best_index]
        if verbose:
            fmt = "fitlen=%d. best fit: x=%d, y=%d, xvy=%d, " \
                "ss=%0.3g, poly=%0.4f %0.4f"
            print fmt % (fitlen, x_midslice_start - fitlen, best_index, 
                x_midslice_start - fitlen - best_index, 
                best_ss / len(chosen_idxs), best_fitpoly[0], best_fitpoly[1])

        # Increase the size
        fitlen = fitlen + 1    
    
    return best_fitpoly

def get_95prctl_r_minus_b(frame):
    """Gets the 95th percentile of the distr of Red - Blue in the frame
    
    Spatially downsample by 2x in x and y to save time.
    """
    vals = (
        frame[::2, ::2, 0].astype(np.int) - 
        frame[::2, ::2, 2].astype(np.int)).flatten()
    return np.sort(vals)[int(.95 * len(vals))]

def get_or_save_lums(session, lumdir=None, meth='gray', verbose=True, 
    image_w=320, image_h=240):
    """Load lum for session from video or if available from cache
    
    Sends kwargs to my.video.process_chunks_of_video
    """    
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
    if verbose:
        print "calculating lums.."
    if meth == 'gray':
        lums = my.video.process_chunks_of_video(vfilename, n_frames=np.inf,
            verbose=verbose, image_w=image_w, image_h=image_h)
    elif meth == 'r-b':
        lums = my.video.process_chunks_of_video(vfilename, n_frames=np.inf,
            func=get_95prctl_r_minus_b, pix_fmt='rgb24', verbose=verbose)
    
    # Save
    my.misc.pickle_dump(lums, new_lum_filename)
    
    return lums
    

def autosync_behavior_and_video_with_houselight(session, save_result=True,
    light_delta=30, diffsize=2, refrac=50, verbose=False, 
    image_w=320, image_h=240):
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
    bfile = session_row['filename']

    # Get the lums ... this takes a while
    # We cache it here so that sync_video_with_behavior doesn't have to
    lums = get_or_save_lums(session, verbose=verbose, 
        image_w=image_w, image_h=image_h)

    # Fit the data
    b2v_fit = sync_video_with_behavior(bfile, lums=lums, video_file=vfilename,
        light_delta=light_delta, diffsize=diffsize, refrac=refrac)
    if b2v_fit is None:
        print "no fit found"

    # Store
    if save_result:
        if b2v_fit is not None:
            BeWatch.db.set_manual_bv_sync(session, b2v_fit)

    return b2v_fit


def autosync_behavior_and_video_with_houselight_from_day(date=None, **kwargs):
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
            autosync_behavior_and_video_with_houselight(session, **kwargs)

def sync_video_with_behavior(bfile, lums=None, video_file=None,
    stop_after_frame=np.inf,
    light_delta=75, diffsize=2, refrac=50,
    assumed_fps=30., error_if_no_fit=False, verbose=False):
    """Sync video with behavioral file
    
    Uses decrements in luminance and the backlight signal to do the sync.
    Assumes the backlight decrement is at the time of entry to state 1.
    
    The luminance signal will be inverted in order to detect decrements.
    Assumes video frame rates is 30fps, regardless of actual frame rate.
    And fits the behavior to the video based on that.
    
    bfile : behavior log, used to extract state change times
    lums : luminances by frame, if pre-calculated
    video_file : if lums is None, then calculates them using this
        video_file
    error_if_no_fit : if True and no fit is found, raises Exception
    verbose : passed to process_chunks_of_video to print out frame
        number for each chunk
    
    See BeWatch.syncing.extract_onsets_and_durations for details on
    light_delta, diffsize, and refrac.
    """    
    # Get the mean luminances
    # Would this be significantly faster if we spatially downsampled?
    # Or used ffmpeg's native calculations?
    if lums is None:
        print "loading luminances ... this will take a while"    
        lums = my.video.process_chunks_of_video(video_file, 
            n_frames=stop_after_frame, verbose=verbose)

    # Get onsets and durations
    onsets, durations = extract_onsets_and_durations(-lums, 
        delta=light_delta, diffsize=diffsize, refrac=refrac)

    # Convert to seconds in the spurious timebase
    v_onsets = onsets / assumed_fps

    # Get the data from Ardulines
    lines = ArduFSM.TrialSpeak.read_lines_from_file(bfile)
    parsed_df_by_trial = \
        ArduFSM.TrialSpeak.parse_lines_into_df_split_by_trial(lines)

    # Find the time of transition into state 1
    backlight_times = ArduFSM.TrialSpeak.identify_state_change_times(
        parsed_df_by_trial, state1=1, show_warnings=True)

    # Find the fit
    b2v_fit = longest_unique_fit(v_onsets, backlight_times)    
    if b2v_fit is None and error_if_no_fit:
        raise ValueError("no fit found")
    return b2v_fit
