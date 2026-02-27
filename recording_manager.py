"""EEG recording management object with EDF/BDF I/O via pyedflib."""

import warnings

import os
import numpy as np
import pandas as pd
import pyedflib
from tqdm import tqdm

class RecordingHandler:

    def __init__(self, eeg_path, log_path, trigger_column='trigger', drop_initial_event=True, trigger_channel_name='Status'):

        self.logpath = os.path.abspath(log_path)
        self.bdfpath = os.path.abspath(eeg_path)

        self.log = self.read_log(self.logpath)

        self.reader = pyedflib.EdfReader(self.eeg_path)
        self.header = self.reader.getHeader()
        self.ch_headers = self.reader.getSignalHeaders()
        self.nsamples = self.reader.samples_in_file(0)
        self.channels = self.reader.getSignalLabels()

        self.digital = True
        filetype='BDF'

        self.trigger_channel = self.read_data(channels=[trigger_channel_name])        
        if filetype == 'BDF':
            self.trigger_channel = self.trigger_channel & 0xFFFF
            self.encodingdict = dict(zip(self.trigger_channel, self.read_data(channels=[trigger_channel_name])))
        self.trigger_default = np.mean(self.trigger_channel)


    def read_data(self, channels=None):

        data = np.zeros((len(channels), self.nsamples,)).astype(np.float32)
        ch_idcs = [self.channels.index(ch) for ch in channels]
        for idx in tqdm(range(len(ch_idcs))):
            data[idx, :] = self.reader.readSignal(idx, digital=self.digital)

    def read_log(self, log_path):
        # infer schema?
        self.log = pd.read_csv(log_path)

    def update_log(self):
        self.log_triggers = self.log[self.triggercol].to_numpy()
        self.unique_log_triggers = np.unique(self.log_triggers)
        self.trigger_events = dict(
            self.log[[self.triggercol, self.eventcol]].drop_duplicates().values
        )
        self.log.reset_index(drop=True)

    def get_eeg_events(self, drop_initial_event=True, minlength=0, verbose=True):

        self.update_log()

        if drop_initial_event:
            first_val = self.trigger_channel[0]
            self.trigger_channel[self.trigger_channel == first_val] = self.trigger_default

        self.unique_eeg_triggers = np.unique(self.trigger_channel)
        self.unique_eeg_triggers = np.delete(
            self.unique_eeg_triggers,
            np.where(self.unique_eeg_triggers == self.trigger_default),
        )

        mask = self.trigger_channel > 0
        boundaries = np.flatnonzero(np.diff(self.trigger_channel) != 0) + 1
        runs = np.split(np.arange(self.trigger_channel.size), boundaries)
        self.bdf_trigger_samples = [
            r for r in runs if mask[r[0]] and len(r) > minlength
        ]
        self.bdf_triggers = np.concatenate([
            np.unique(self.trigger_channel[e]) for e in self.bdf_trigger_samples
        ])
        self.trigger_times = np.asarray([
            t[0] / self.sfreq for t in self.bdf_trigger_samples
        ])
        ### figure out order of verbose and warnings - do we want something bringing log and eeg together - or is this fine?
        if verbose:
            print(
                f'Found {len(self.bdf_triggers)} events in data file'
                f'with unique triggers: {self.unique_eeg_triggers}.'
            )

        if len(self.log_triggers) != len(self.bdf_triggers):
            warnings.warn(
                'Uneven number of events in log and BDF file; '
                'risk of timing discrepancies. Please inspect before proceeding.',
                RuntimeWarning,
            )
        elif np.all(self.log_triggers == self.bdf_triggers):
            self.log['chan_idcs'] = self.bdf_trigger_samples
        else:
            warnings.warn(
                'Log and BDF trigger codes do not match; '
                'risk of timing discrepancies. Please inspect before proceeding.',
                RuntimeWarning,
            )

    def find_event(self, lookup, mode='or', asint=True):

        mask = pd.DataFrame(np.full(self.log.shape, True),
                            columns = self.log.columns)
        
        for key, vals in lookup.items():
            mask[key] &= self.log[key].isin(vals)

        mask = mask.to_numpy()
        if mode == 'or':
            mask = np.any(mask, axis=1)
        elif mode == 'and':
            mask = np.all(mask, axis=1)

        return np.where(mask)[0] if asint else mask

    def insert_log_event(self, indices, events):

        if isinstance(events, dict):
            events = pd.DataFrame(events)

        for column in self.log.columns:
            if column not in events.columns:
                events[column] = np.nan

        insert_arr = pd.DataFrame(events).to_numpy()
        lognp = self.log.to_numpy()
        self.log = pd.DataFrame(
            np.insert(lognp, indices, insert_arr, axis=0),
            columns=self.log.columns,
        )
        self.update_log()

    def insert_eeg_triggers(self, triggers, times=None, samples=None,
                            trigger_span=15, allow_overlap=False):

        if samples is None:
            samples = np.round(self.sfreq * np.asarray(times)).astype(int)

        if np.isscalar(samples[0]):
            samples = [np.arange(s, s + trigger_span) for s in samples]

        if isinstance(triggers, list) and np.isscalar(triggers[0]):
            triggers = [[t] * len(samples[idx]) for idx, t in enumerate(triggers)]

        samples_flat = np.concatenate(samples)
        triggers_flat = np.concatenate(triggers)

        if not allow_overlap and np.any(
            np.isin(self.trigger_channel[samples_flat], self.unique_eeg_triggers)
        ):
            raise ValueError('Triggers already exist at the requested time points.')

        self.trigger_channel[samples_flat] = triggers_flat

    def run_diagnostic(self):

        pass