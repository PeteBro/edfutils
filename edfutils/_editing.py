"""Editing mixin for EEGSession — event extraction, insertion, and data modification."""

import copy
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd


class _EditingMixin:

    def update_log(self):
        """Refresh log-derived attributes from the current state of ``self.log``.

        Updates ``log_triggers``, ``log_times``, and ``unique_log_triggers``,
        and resets the log index.
        """
        self.log_triggers = self.log[self.trigger_column].to_numpy()
        self.log_times = self.log[self.onset_column].to_numpy()
        self.unique_log_triggers = np.unique(self.log_triggers)

        self.log = self.log.reset_index(drop=True)


    def get_eeg_events(self, drop_initial_event=True, minlength=0, verbose=True):
        """Extract trigger events from the EEG trigger channel and align with the log.

        Populates ``eeg_trigger_samples``, ``eeg_triggers``, ``trigger_times``,
        and ``unique_eeg_triggers``. If the log and EEG triggers match,
        ``trigger_idcs`` is added to ``self.log``. Warns on count or code mismatches.

        Args:
            drop_initial_event (bool): If True, replace the first trigger value
                (typically a recording-start artefact) with the baseline.
            minlength (int): Minimum run length (in samples) for a trigger to
                be included.
            verbose (bool): If True, print a summary of found events.
        """
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
        self.eeg_trigger_samples = [
            r for r in runs if mask[r[0]] and len(r) > minlength
        ]
        self.eeg_triggers = np.concatenate([
            np.unique(self.trigger_channel[e]) for e in self.eeg_trigger_samples
        ])
        self.trigger_times = np.asarray([
            t[0] / self.sfreq for t in self.eeg_trigger_samples
        ])

        if verbose:
            print(
                f'Found {len(self.eeg_triggers)} events in data file '
                f'with unique triggers: {self.unique_eeg_triggers}.'
            )

        if len(self.log_triggers) != len(self.eeg_triggers):
            warnings.warn(
                'Uneven number of events in log and BDF file; '
                'risk of timing discrepancies. Please run diagnostics before proceeding.',
                RuntimeWarning,
            )

        elif not np.all(self.log_triggers == self.eeg_triggers):
            warnings.warn(
                'Log and BDF trigger codes do not match; '
                'risk of timing discrepancies. Please run diagnostics before proceeding.',
                RuntimeWarning,
            )

        else:
            self.log['trigger_idcs'] = self.eeg_trigger_samples


    def find_event(self, lookup, mode='and', asint=True):
        """Find log row indices matching a set of column-value filters.

        Args:
            lookup (dict): Mapping of column name to value or list of values.
            mode (str): ``'or'`` to match any filter, ``'and'`` to match all.
            asint (bool): If True, return integer indices; otherwise return a
                boolean mask.

        Returns:
            np.ndarray: Matching row indices or boolean mask.
        """
        mask = defaultdict(list)

        for key, vals in lookup.items():
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            mask[key] = self.log[key].isin(vals).to_list()

        mask = pd.DataFrame(mask).to_numpy()

        if mode == 'or':
            mask = np.any(mask, axis=1)
        elif mode == 'and':
            mask = np.all(mask, axis=1)

        return np.where(mask)[0] if asint else mask
    

    def modify_event(self):
        #TODO; Needs adding - modify event, triggers too maybe?
        raise NotImplementedError


    def _insert_log_event(self, indices, events):
        """Insert one or more rows into the event log at the given indices.

        Missing columns are filled with NaN. Calls ``update_log`` afterwards.

        Args:
            indices (int or array-like): Row position(s) at which to insert.
            events (dict or pd.DataFrame): Event data to insert.
        """
        if isinstance(events, dict):
            events = pd.DataFrame(events)

        for column in self.log.columns:
            if column not in events.columns:
                events[column] = np.nan
        
        events = events[self.log.columns]

        insert_arr = pd.DataFrame(events).to_numpy()
        lognp = self.log.to_numpy()
        self.log = pd.DataFrame(
            np.insert(lognp, indices, insert_arr, axis=0),
            columns=self.log.columns,
        )
        self.update_log()


    def _insert_eeg_triggers(self, triggers, times=None, samples=None,
                            trigger_span=15, allow_overlap=False):
        """Write trigger codes directly into the EEG trigger channel.

        Either ``times`` or ``samples`` must be provided. Scalar sample values
        are expanded into spans of ``trigger_span`` samples.

        Args:
            triggers (list): Trigger code(s) to write.
            times (array-like, optional): Event onset times in seconds.
            samples (array-like, optional): Event onset sample indices (or
                pre-built arrays of sample indices per event).
            trigger_span (int): Number of samples each trigger occupies when
                expanding scalar sample positions.
            allow_overlap (bool): If False, raises ValueError if any target
                samples already contain a known trigger code.

        Raises:
            ValueError: If ``allow_overlap`` is False and triggers already
                exist at the requested positions.
        """
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


    def insert_event(self, events, trigger_span=15, allow_overlap=False):
        """Insert one or more events into the recording.

        Adds the events to the log and writes the corresponding trigger codes
        into the EEG data at the correct time points.

        Args:
            events (dict or pd.DataFrame): Event data; must include onset
                times and trigger codes.
            trigger_span (int): Duration of each trigger code in samples.
            allow_overlap (bool): If False, raises an error if triggers already
                exist at the target time points.
        """

        if isinstance(events, dict):
            events = pd.DataFrame(events)

        times = events[self.onset_column].to_numpy()
        indices = np.searchsorted(self.log.onset.to_numpy(), times)
        self._insert_log_event(indices, events)
        
        last_event = np.max(np.where(~self.log['trigger_idcs'].isna()))
        last_event = self.log.iloc[last_event]
        tdiffs = [last_event.onset - self.log.iloc[idx].onset for idx in indices]

        ref_samples = np.asarray(last_event.trigger_idcs)
        insert_samples = [ref_samples -  int(tdiff*self.sfreq) for tdiff in tdiffs]        

        self._insert_eeg_triggers(
            list(events[self.trigger_column]),
            samples=insert_samples,
            trigger_span=trigger_span,
            allow_overlap=allow_overlap,
        )
        
        self.get_eeg_events(verbose=False)


    def insert_span(self, start, stop=None, length=None, value=0):
        """Insert a constant-value span of samples into the signal data.

        Args:
            start (float): Insertion point in seconds.
            stop (float, optional): End of the span in seconds. Used to derive
                ``length`` if not provided directly.
            length (int, optional): Number of samples to insert.
            value (int): Fill value for the inserted samples.
        """
        if length is None:
            length = int((stop - start) * self.sfreq)

        start_sample = int(start * self.sfreq)

        span = np.full((self._data.shape[0], length), value, dtype=self.dtype)
        self._data = np.concatenate(
            [self._data[:, :start_sample], span, self._data[:, start_sample:]],
            axis=1,
        )

        self.get_eeg_events(verbose=False)


    def drop_channels(self, channels):
        """Remove channels from the loaded data.

        Args:
            channels (list[str]): Names of channels to drop.
        """
        for ch in channels:
            self.channel.drop(ch)


    def crop(self, start=None, stop=None, tmin=None, tmax=None, inplace=False):
        """Crop the recording to a time window or log row range.

        Two modes are supported:

        - **Time-based** (``tmin``/``tmax``): selects log rows whose onset
          falls within ``[tmin, tmax]`` and pads the data slice to cover the
          full requested interval.
        - **Index-based** (``start``/``stop``): selects log rows
          ``start:stop`` with no padding.

        Args:
            start (int, optional): Start log row index (index-based mode).
            stop (int, optional): Stop log row index, exclusive (index-based mode).
            tmin (float, optional): Start time in seconds (time-based mode).
            tmax (float, optional): Stop time in seconds (time-based mode).
            inplace (bool): If True, modify this object in place; otherwise
                return a deep copy with the cropped data.

        Returns:
            EEGSession or None: Cropped copy if ``inplace=False``, else None.
        """

        if tmin is not None and tmax is not None:
            query = self.log[self.onset_column].to_numpy()
            indices = np.where((query >= tmin) & (query <= tmax))[0]
            onset_times = self.log.loc[indices, self.onset_column].to_numpy()
            start_diff = onset_times[0] - tmin
            stop_diff = tmax - onset_times[-1]
        else:
            indices = np.arange(start, stop)
            start_diff = stop_diff = 0

        cropped_log = self.log.loc[indices].copy()

        bdfspan = np.concatenate(cropped_log['trigger_idcs'].to_list())
        bdfstart = bdfspan[0] - int(start_diff * self.sfreq)
        bdfstop = bdfspan[-1] + int(stop_diff * self.sfreq)

        cropped_channels = self._data[:, bdfstart:bdfstop].copy()

        if inplace:
            self._data = cropped_channels
            self.log = cropped_log
            self.get_eeg_events()

        else:
            new = copy.deepcopy(self)
            new._data = cropped_channels
            new.log = cropped_log
            new.get_eeg_events()
            return new
