"""Module for generating overlays of shape positions"""
import os
import numpy as np
import pandas
import ArduFSM
import scipy.misc
import my
from ArduFSM import TrialMatrix, TrialSpeak, mainloop
import matplotlib.pyplot as plt
import my.plot
import BeWatch

def load_frames_by_trial(frame_dir, trials_info):
    """Read all trial%03d.png in frame_dir and return as dict"""
    trialnum2frame = {}
    for trialnum in trials_info.index:
        filename = os.path.join(frame_dir, 'trial%03d.png' % trialnum)
        if os.path.exists(filename):
            im = scipy.misc.imread(filename, flatten=True)
            trialnum2frame[trialnum] = im    
    return trialnum2frame

def mean_frames_by_choice(trials_info, trialnum2frame):
    # Keep only those trials that we found images for
    trials_info = trials_info.ix[sorted(trialnum2frame.keys())]

    # Dump those with a spoiled trial
    trials_info = my.misc.pick_rows(trials_info, choice=[0, 1], bad=False)

    # Split on choice
    res = []
    gobj = trials_info.groupby('choice')
    for choice, subti in gobj:
        meaned = np.mean([trialnum2frame[trialnum] for trialnum in subti.index],
            axis=0)
        res.append({'choice': choice, 'meaned': meaned})
    resdf_choice = pandas.DataFrame.from_records(res)

    return resdf_choice

def calculate_performance(trials_info, p_servothrow):
    """Use p_servothrow to calculate performance by stim number"""
    rec_l = []
    
    # Assign pos_delta
    raw_servo_positions = np.unique(trials_info.servo_position)
    if len(raw_servo_positions) == 1:
        pos_delta = 25
    else:
        pos_delta = raw_servo_positions[1] - raw_servo_positions[0]
    p_servothrow.pos_delta = pos_delta
    
    # Convert
    ti2 = p_servothrow.assign_trial_type_to_trials_info(trials_info)
    
    # Perf by ST and by SN
    gobj = ti2.groupby(['rewside', 'servo_intpos', 'stim_number'])
    for (rewside, servo_intpos, stim_number), sub_ti in gobj:
        nhits, ntots = ArduFSM.trials_info_tools.calculate_nhit_ntot(sub_ti)
        if ntots > 0:
            rec_l.append({
                'rewside': rewside, 'servo_intpos': servo_intpos,
                'stim_number': stim_number,
                'perf': nhits / float(ntots),
                })

    # Form dataframe
    df = pandas.DataFrame.from_records(rec_l)
    return df



def plot_side_perf(ax, perf):
    """Plot performance on each side vs servo position"""
    colors = ['b', 'r']
    for rewside in [0, 1]:
        # Form 2d perf matrix for this side by unstacking
        sideperf = perf[rewside].unstack() # servo on rows, stimnum on cols
        yvals = map(int, sideperf.index)
        
        # Mean over stim numbers
        meaned_sideperf = sideperf.mean(axis=0)
        
        # Plot
        ax.plot(yvals, sideperf.mean(axis=1), color=colors[rewside])
    
    # Avg over sides
    meaned = perf.unstack(1).mean()
    ax.plot(yvals, meaned, color='k')
    
    ax.set_xlabel('servo position')
    ax.set_ylim((0, 1))
    ax.set_yticks((0, .5, 1))
    ax.set_xticks(yvals) # because on rows
    
    ax.plot(ax.get_xlim(), [.5, .5], 'k:')


def make_overlay(sess_meaned_frames, ax, meth='all'):
    """Plot various overlays
    
    sess_meaned_frames - df with columns 'meaned', 'rewside', 'servo_pos'
    meth -
        'all' - average all with the same rewside together
        'L' - take closest L and furthest R
        'R' - take furthest R and closest L
        'close' - take closest of both
        'far' - take furthest of both
    """
    # Split into L and R
    if meth == 'all':
        L = np.mean(sess_meaned_frames['meaned'][
            sess_meaned_frames.rewside == 'left'], axis=0)
        R = np.mean(sess_meaned_frames['meaned'][
            sess_meaned_frames.rewside == 'right'], axis=0)
    elif meth == 'L':
        closest_L = my.pick_rows(sess_meaned_frames, rewside='left')[
            'servo_pos'].min()
        furthest_R = my.pick_rows(sess_meaned_frames, rewside='right')[
            'servo_pos'].max()
        L = my.pick_rows(sess_meaned_frames, rewside='left', 
            servo_pos=closest_L).irow(0)['meaned']
        R = my.pick_rows(sess_meaned_frames, rewside='right', 
            servo_pos=furthest_R).irow(0)['meaned']
    elif meth == 'R':
        closest_R = my.pick_rows(sess_meaned_frames, rewside='right')[
            'servo_pos'].min()
        furthest_L = my.pick_rows(sess_meaned_frames, rewside='left')[
            'servo_pos'].max()
        L = my.pick_rows(sess_meaned_frames, rewside='left', 
            servo_pos=furthest_L).irow(0)['meaned']
        R = my.pick_rows(sess_meaned_frames, rewside='right', 
            servo_pos=closest_R).irow(0)['meaned']     
    elif meth == 'close':
        closest_L = my.pick_rows(sess_meaned_frames, rewside='left')[
            'servo_pos'].min()
        closest_R = my.pick_rows(sess_meaned_frames, rewside='right')[
            'servo_pos'].min()
        L = my.pick_rows(sess_meaned_frames, rewside='left', 
            servo_pos=closest_L).irow(0)['meaned']
        R = my.pick_rows(sess_meaned_frames, rewside='right', 
            servo_pos=closest_R).irow(0)['meaned']     
    elif meth == 'far':
        furthest_L = my.pick_rows(sess_meaned_frames, rewside='left')[
            'servo_pos'].max()            
        furthest_R = my.pick_rows(sess_meaned_frames, rewside='right')[
            'servo_pos'].max()
        L = my.pick_rows(sess_meaned_frames, rewside='left', 
            servo_pos=furthest_L).irow(0)['meaned']
        R = my.pick_rows(sess_meaned_frames, rewside='right', 
            servo_pos=furthest_R).irow(0)['meaned']     
    else:
        raise ValueError("meth not understood: " + str(meth))
            
    # Color them into the R and G space, with zeros for B
    C = np.array([L, R, np.zeros_like(L)])
    C = C.swapaxes(0, 2).swapaxes(0, 1) / 255.

    my.plot.imshow(C, ax=ax, axis_call='image', origin='upper')
    ax.set_xticks([]); ax.set_yticks([])

    return C


def cached_dump_frames_at_retraction_times(rows, frame_dir='./frames'):
    """Wrapper around dump_frames_at_retraction_time
    
    Repeats call for each row in rows, as long as the subdir doesn't exist.
    """
    if not os.path.exists(frame_dir):
        print "auto-creating", frame_dir
        os.mkdir(frame_dir)

    # Iterate over sessions
    for idx in rows.index:
        # Something very strange here where iterrows distorts the dtype
        # of the object arrays
        row = rows.ix[idx]

        # Set up output_dir and continue if already exists
        output_dir = os.path.join(frame_dir, row['behave_filename'])
        if os.path.exists(output_dir):
            continue
        else:
            print "auto-creating", output_dir
            os.mkdir(output_dir)
            print output_dir

        # Dump the frames
        dump_frames_at_retraction_time(row, session_dir=output_dir)



def generate_meaned_frames(session):
    """Generates the 'sess_meaned_frames', split by side and servo.
    """
    # Load data
    sbvdf = BeWatch.db.get_synced_behavior_and_video_df()
    msdf = BeWatch.db.get_manual_sync_df()
    PATHS = BeWatch.db.get_paths()

    # Join all the dataframes we need and check that session is in there
    jdf = sbvdf.join(msdf, on='session', how='right')
    metadata = jdf[jdf.session == session]
    if len(metadata) != 1:
        raise ValueError("session %s not found for overlays" % session)
    metadata = metadata.irow(0)
    
    # Dump the frames
    frame_dir = os.path.join(PATHS['database_root'], 'frames', session)
    if not os.path.exists(frame_dir):
        raise ValueError("no frames for %s, run make_overlays_for_all_fits")

    # Reload the frames
    trial_matrix = BeWatch.db.get_trial_matrix(session)
    trialnum2frame = load_frames_by_trial(frame_dir, trial_matrix)

    # Keep only those trials that we found images for
    trial_matrix = trial_matrix.ix[sorted(trialnum2frame.keys())]

    # Split on side, servo_pos, stim_number
    res = []
    gobj = trial_matrix.groupby(['rewside', 'servo_pos', 'stepper_pos'])
    for (rewside, servo_pos, stim_number), subti in gobj:
        meaned = np.mean([trialnum2frame[trialnum] for trialnum in subti.index],
            axis=0)
        res.append({'rewside': rewside, 'servo_pos': servo_pos, 
            'stim_number': stim_number, 'meaned': meaned})
    resdf = pandas.DataFrame.from_records(res)    
    
    return resdf

def timedelta_to_seconds1(val):
    """Often it ends up as a 0d timedelta array.
    
    This especially happens when taking a single row from a df, which becomes
    a series. Then you sometimes cannot divide by np.timedelta64(1, 's')
    or by 1e9
    """
    ite = val.item() # in nanoseconds
    return ite / 1e9

def timedelta_to_seconds2(val):
    """More preferred ... might have been broken in old versions."""
    return val / np.timedelta64(1, 's')

def make_overlays_from_all_fits(overwrite_frames=False, savefig=True):
    """Makes overlays for all available sessions"""
    # Load data
    sbvdf = BeWatch.db.get_synced_behavior_and_video_df()
    msdf = BeWatch.db.get_manual_sync_df()
    
    # Join all the dataframes we need
    jdf = sbvdf.join(msdf, on='session', how='right')

    # Do each
    for session in jdf.session:
        make_overlays_from_fits(session, overwrite_frames=overwrite_frames,
            savefig=savefig)

def make_overlays_from_fits(session, overwrite_frames=False, savefig=True):
    """Given a session name, generates overlays.

    If savefig: then it will save the figure in behavior_db/overlays
        However, if that file already exists, it will exist immediately.
    If overwrite_frames: then it will always redump the frames
    """
    # Load data
    sbvdf = BeWatch.db.get_synced_behavior_and_video_df()
    msdf = BeWatch.db.get_manual_sync_df()
    PATHS = BeWatch.db.get_paths()

    # Choose the savename and skip if it exists
    if savefig:
        savename = os.path.join(PATHS['database_root'], 'overlays',
            session + '.png')
        if os.path.exists(savename):
            return
    print session
    
    # Join all the dataframes we need and check that session is in there
    jdf = sbvdf.join(msdf, on='session', how='right')
    metadata = jdf[jdf.session == session]
    if len(metadata) != 1:
        raise ValueError("session %s not found for overlays" % session)
    metadata = metadata.irow(0)
    
    # Dump the frames
    frame_dir = os.path.join(PATHS['database_root'], 'frames', session)
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)
        dump_frames_at_retraction_time(metadata, frame_dir)
    elif overwrite_frames:
        dump_frames_at_retraction_time(metadata, frame_dir)

    # Reload the frames
    trial_matrix = BeWatch.db.get_trial_matrix(session)
    trialnum2frame = load_frames_by_trial(frame_dir, trial_matrix)

    # Keep only those trials that we found images for
    trial_matrix = trial_matrix.ix[sorted(trialnum2frame.keys())]

    # Split on side, servo_pos, stim_number
    res = []
    gobj = trial_matrix.groupby(['rewside', 'servo_pos', 'stepper_pos'])
    for (rewside, servo_pos, stim_number), subti in gobj:
        meaned = np.mean([trialnum2frame[trialnum] for trialnum in subti.index],
            axis=0)
        res.append({'rewside': rewside, 'servo_pos': servo_pos, 
            'stim_number': stim_number, 'meaned': meaned})
    resdf = pandas.DataFrame.from_records(res)

    # Make the various overlays
    f, axa = plt.subplots(2, 3, figsize=(13, 6))
    make_overlay(resdf, axa[0, 0], meth='all')
    make_overlay(resdf, axa[1, 1], meth='L')
    make_overlay(resdf, axa[1, 2], meth='R')
    make_overlay(resdf, axa[0, 1], meth='close')
    make_overlay(resdf, axa[0, 2], meth='far')
    f.suptitle(session)
    
    # Save or show
    if savefig:
        f.savefig(savename)
        plt.close(f)
    else:
        plt.show()    


def dump_frames_at_retraction_time(metadata, session_dir):
    """Dump the retraction time frame into a subdirectory.
    
    metadata : row containing behavior info, video info, and fit info    
    """
    # Load trials info
    trials_info = TrialMatrix.make_trial_matrix_from_file(metadata['filename'])
    splines = TrialSpeak.load_splines_from_file(metadata['filename'])

    # Insert servo retract time
    lines = TrialSpeak.read_lines_from_file(metadata['filename'])
    parsed_df_split_by_trial = \
        TrialSpeak.parse_lines_into_df_split_by_trial(lines)    
    trials_info['time_retract'] = TrialSpeak.identify_servo_retract_times(
        parsed_df_split_by_trial)        

    # Fit to video times
    fit = metadata['fit0'], metadata['fit1']
    video_times = trials_info['time_retract'].values - \
        metadata['guess_vvsb_start']
    trials_info['time_retract_vbase'] = np.polyval(fit, video_times)
    
    # Mask out any frametimes that are before or after the video
    duration_s = timedelta_to_seconds2(metadata['duration_video'])
    BeWatch.syncing.mask_by_buffer_from_end(trials_info['time_retract_vbase'], 
        end_time=duration_s, buffer=10)
    
    # Dump frames
    frametimes_to_dump = trials_info['time_retract_vbase'].dropna()
    for trialnum, frametime in trials_info['time_retract_vbase'].dropna().iterkv():
        output_filename = os.path.join(session_dir, 'trial%03d.png' % trialnum)
        my.video.frame_dump(metadata['filename_video'], 
            frametime, meth='ffmpeg fast',
            output_filename=output_filename)


