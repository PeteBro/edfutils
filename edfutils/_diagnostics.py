"""Diagnostics and repair mixin for EEGSession."""

import warnings

import numpy as np
import pandas as pd


#class _DiagnosticsMixin:
#    """Diagnostics and targeted repairs for EEG/log alignment issues."""
    
import os
import glob
import numpy as np

from edfutils import EEGSession

DATA = '/home/peter/Downloads/'#'/media/peter/WD/s1000'
sub = 'subj01'
ses = 'session-03'

edf_path = glob.glob(os.path.join(DATA, sub, ses, '*.bdf'))[0]
log_path = glob.glob(os.path.join(DATA, sub, ses, '*log*'))[0]

session = EEGSession(edf_path, log_path)

session.get_eeg_events()
session.read_data(['A1'])


def _compare_trigger_count():

    edf_count = dict(zip(*np.unique(session.eeg_triggers, return_counts=True)))
    log_count = dict(zip(*np.unique(session.log_triggers, return_counts=True)))

    triggers = sorted(set(edf_count) | set(log_count))
    result = {
        int(trigger): {
            'log': int(log_count.get(trigger, 0)),
            'edf': int(edf_count.get(trigger, 0)),
            'diff': int(edf_count.get(trigger, 0) - log_count.get(trigger, 0)),
        }
        for trigger in triggers
    }

    return result


def _compare_trigger_order():

    n_log = len(session.log_triggers)
    n_edf = len(session.eeg_triggers)
    if n_log != n_edf:
        raise ValueError(
            f'Cannot compare trigger order with unequal lengths: '
            f'log={n_log}, edf={n_edf}.'
        )

    match = session.log_triggers == session.eeg_triggers
    result = {
        'match': match,
        'mismatch': np.where(~match)[0],
        'n_compared': int(n_log),
        'n_log': int(n_log),
        'n_edf': int(n_edf),
    }

    return result
    
    
def _check_time_mismatch(tolerance=0.1):

    n_compared = min(len(session.log_times), len(session.trigger_times))
    if n_compared == 0:
        tdiff = np.asarray([], dtype=float)
    else:
        edf_time = session.trigger_times[:n_compared].copy()
        log_time = session.log_times[:n_compared].copy()
        edf_time -= edf_time[0]
        log_time -= log_time[0]
        tdiff = edf_time - log_time

    mismatch = np.where(np.abs(tdiff) > tolerance)[0]
    result = {
        'tdiff': tdiff,
        'mismatch': mismatch,
        'tolerance': float(tolerance),
        'n_compared': int(n_compared),
        'n_log': int(len(session.log_times)),
        'n_edf': int(len(session.trigger_times)),
    }

    return result
    


def _check_trigger_pulses(minlen=1, maxlen=None):

    lengths = np.asarray([len(span) for span in session.eeg_trigger_samples])
    too_short = lengths < minlen
    too_long = np.zeros(len(lengths), dtype=bool) if maxlen is None else lengths > maxlen
    merged = np.asarray([
        len(np.unique(session.trigger_channel[span])) > 1
        for span in session.eeg_trigger_samples
    ])

    problem = too_short | too_long | merged
    result = {
        'lengths': lengths,
        'too_short': too_short,
        'too_long': too_long,
        'merged': merged,
        'problem': problem,
        'problem_indices': np.where(problem)[0],
        'minlen': int(minlen),
        'maxlen': None if maxlen is None else int(maxlen),
    }

    return result


def _check_orphaned_triggers(window=10):

    log_triggers = session.log_triggers
    edf_triggers = session.eeg_triggers
    n_compared = min(len(log_triggers), len(edf_triggers))
    mismatch = np.where(log_triggers[:n_compared] != edf_triggers[:n_compared])[0]

    orphan_log = np.arange(n_compared, len(log_triggers))
    orphan_edf = np.arange(n_compared, len(edf_triggers))
    first_mismatch = None if len(mismatch) == 0 else int(mismatch[0])

    result = {
        'first_mismatch': first_mismatch,
        'mismatch': mismatch,
        'orphan_log': orphan_log,
        'orphan_edf': orphan_edf,
        'orphan_log_triggers': log_triggers[orphan_log],
        'orphan_edf_triggers': edf_triggers[orphan_edf],
        'n_compared': int(n_compared),
        'n_log': int(len(log_triggers)),
        'n_edf': int(len(edf_triggers)),
    }

    if first_mismatch is None:
        return result

    log_start = max(0, first_mismatch - window)
    log_stop = min(len(log_triggers), first_mismatch + window + 1)
    edf_start = max(0, first_mismatch - window)
    edf_stop = min(len(edf_triggers), first_mismatch + window + 1)

    result.update({
        'log_window_indices': np.arange(log_start, log_stop),
        'edf_window_indices': np.arange(edf_start, edf_stop),
        'log_window_triggers': log_triggers[log_start:log_stop],
        'edf_window_triggers': edf_triggers[edf_start:edf_stop],
    })
    return result
