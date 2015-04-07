""""Module for miscellaneous BeWatch stuff"""
import BeWatch
import ArduFSM
import numpy as np

def get_choice_times(behavior_filename, verbose=False):
    """Calculates the choice time for each trial in the logfile"""
    # Find the state number for response window
    state_num2names = BeWatch.db.get_state_num2names()    
    resp_win_num = dict([(v, k) for k, v in state_num2names.items()])[
        'RESPONSE_WINDOW']
    
    # Get the lines
    lines = ArduFSM.TrialSpeak.read_lines_from_file(behavior_filename)
    parsed_df_by_trial = \
        ArduFSM.TrialSpeak.parse_lines_into_df_split_by_trial(lines, 
            verbose=verbose)
    
    # Identify times of state change out of response window
    choice_times = ArduFSM.TrialSpeak.identify_state_change_times(
        parsed_df_by_trial, state0=resp_win_num)
    
    return choice_times    