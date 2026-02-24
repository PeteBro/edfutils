"""EEG session management with EDF/BDF I/O via pyedflib."""

import warnings

import numpy as np
import pandas as pd
import pyedflib
from tqdm import tqdm


class SessionHandler:
    """Manages EEG session data including trigger/event handling and EDF/BDF I/O.

    Parameters
    ----------
    log : pd.DataFrame
        Event log with at least trigger and event columns.
    channels : np.ndarray
        Channel data of shape (n_channels, n_samples).
    channel_headers : list of dict
        Per-channel metadata (pyedflib signal header format).
    header : dict
        File-level metadata (pyedflib header format).
    trigger_ch_name : str
        Label of the status/trigger channel.
    convert_triggers : bool
        If True, mask trigger channel to lower 16 bits on load (BDF encoding).
    default_trigger : int
        Value representing "no trigger" (baseline).
    eventcol : str
        Column name in log for event labels.
    triggercol : str
        Column name in log for trigger codes.
    sfreq : int or None
        Sampling frequency in Hz. Inferred from channel_headers if None.
    drop_initial_event : bool
        If True, replace repeated first-sample trigger value with default.
    """

    def __init__(self, log, channels, channel_headers, header,
                 trigger_ch_name='Status', convert_triggers=True,
                 default_trigger=0, eventcol='event', triggercol='trigger',
                 sfreq=None, drop_initial_event=True):

        self.channels = channels
        self.original_copy = channels.copy()
        self.nsamples = channels.shape[-1]
        self.ch_headers = list(channel_headers)
        self.header = dict(header)
        self.dtype = channels.dtype

        self.trigger_ch_name = trigger_ch_name
        self.get_channels()

        self.trigger_channel = self.channels[self.trigger_ch_idx].copy().squeeze()
        self.trigger_default = default_trigger
        self.trigger_copy = self.trigger_channel.copy()
        self.convert_triggers = convert_triggers

        if self.convert_triggers:
            self.trigger_channel = self.trigger_channel & 0xFFFF

        self.encodingdict = dict(zip(self.trigger_channel, self.trigger_copy))

        self.log = log.copy()
        self.eventcol = eventcol
        self.triggercol = triggercol

        if sfreq is None:
            sfreqs = {c['sample_frequency'] for c in channel_headers}
            if len(sfreqs) == 1:
                self.sfreq = int(list(sfreqs)[0])
            else:
                raise ValueError(
                    'Inconsistent sample rates across channels; '
                    'please specify sfreq explicitly.'
                )
        else:
            self.sfreq = int(sfreq)

        self.samples_to_time()
        self.update_log()
        self.get_events(verbose=True, drop_initial_event=drop_initial_event)

    # ------------------------------------------------------------------ #
    # Constructors / factories                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_edf(cls, path, log, trigger_ch_name='Status', convert_triggers=True,
                 default_trigger=0, eventcol='event', triggercol='trigger',
                 sfreq=None, drop_initial_event=True):
        """Create a SessionHandler by reading an EDF/BDF file with pyedflib.

        Parameters
        ----------
        path : str or path-like
            Path to the EDF/BDF file.
        log : pd.DataFrame
            Event log (see class docstring).
        trigger_ch_name, convert_triggers, default_trigger, eventcol,
        triggercol, sfreq, drop_initial_event :
            Forwarded to ``__init__``.

        Returns
        -------
        SessionHandler
        """
        f = pyedflib.EdfReader(str(path))
        try:
            n = f.signals_in_file
            channel_headers = [f.getSignalHeader(i) for i in range(n)]
            header = f.getHeader()
            n_samples = f.getNSamples()
            max_samples = int(max(n_samples))
            # Load as digital (integer) values so bit operations on the
            # trigger/status channel are correct.
            channels = np.zeros((n, max_samples), dtype=np.int32)
            for i in range(n):
                sig = f.readSignal(i, digital=True)
                channels[i, :len(sig)] = sig
            file_type = f.filetype
        finally:
            f.close()

        obj = cls(log, channels, channel_headers, header,
                  trigger_ch_name=trigger_ch_name,
                  convert_triggers=convert_triggers,
                  default_trigger=default_trigger,
                  eventcol=eventcol,
                  triggercol=triggercol,
                  sfreq=sfreq,
                  drop_initial_event=drop_initial_event)
        obj._file_type = file_type
        return obj

    @classmethod
    def merge_recordings(cls, raw_1, raw_2, header_1, header_2,
                         channel_headers_1, channel_headers_2, log,
                         **session_kwargs):
        """Merge two raw recordings into one SessionHandler.

        Inserts a silence span at the join point to account for any timing
        gap between the two recordings as indicated by the event log.

        Parameters
        ----------
        raw_1, raw_2 : np.ndarray
            Channel arrays (n_channels, n_samples) for each recording.
        header_1, header_2 : dict
            File-level headers for each recording.
        channel_headers_1, channel_headers_2 : list of dict
            Per-channel headers for each recording.
        log : pd.DataFrame
            Combined event log for both recordings.
        **session_kwargs
            Additional kwargs forwarded to ``__init__``.

        Returns
        -------
        SessionHandler
            Merged session (not yet returned as raw data; call return_data()
            or write_edf() when finished).
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            part_1 = cls(log, raw_1, channel_headers_1, header_1, **session_kwargs)
            t1_idx = len(part_1.bdf_triggers) - 1
            raw1_dur = len(part_1.trigger_channel) / part_1.sfreq

            part_2 = cls(log, raw_2, channel_headers_2, header_2, **session_kwargs)
            t2_idx = len(part_2.bdf_triggers)

        raw = np.concatenate([raw_1, raw_2], axis=1)
        session = cls(log, raw, channel_headers_1, header_1, **session_kwargs)

        t1 = session.bdf_trigger_samples[t1_idx][0] / session.sfreq
        t2 = session.bdf_trigger_samples[-t2_idx][0] / session.sfreq
        bdf_tdiff = t2 - t1

        l1 = session.log.iloc[t1_idx]['time']
        l2 = session.log.iloc[-t2_idx]['time']
        log_tdiff = l2 - l1

        time_to_add = log_tdiff - bdf_tdiff
        session.insert_span(start=raw1_dur, stop=raw1_dur + time_to_add)

        session.update_log()
        session.get_events()
        return session

    # ------------------------------------------------------------------ #
    # EDF/BDF I/O                                                          #
    # ------------------------------------------------------------------ #

    def write_edf(self, path):
        """Write session data to an EDF/BDF file using pyedflib.

        The trigger channel is re-encoded before writing.  The file type is
        preserved from the source file when loaded via ``from_edf``; otherwise
        ``pyedflib.FILETYPE_BDFPLUS`` is used as the default.

        Parameters
        ----------
        path : str or path-like
            Destination file path.
        """
        header, ch_headers, channels, _ = self.return_data()
        file_type = getattr(self, '_file_type', pyedflib.FILETYPE_BDFPLUS)
        n_channels = channels.shape[0]

        f = pyedflib.EdfWriter(str(path), n_channels, file_type=file_type)
        try:
            f.setHeader(header)
            f.setSignalHeaders(ch_headers)
            f.writeSamples([channels[i] for i in range(n_channels)])
        finally:
            f.close()

    # ------------------------------------------------------------------ #
    # Core internals                                                       #
    # ------------------------------------------------------------------ #

    def get_channels(self):
        """Populate trigger channel index and channel name/index mappings."""
        self.trigger_ch_idx = int(np.squeeze(
            np.where([c['label'] == self.trigger_ch_name for c in self.ch_headers])
        ))
        self.channel_names = [c['label'] for c in self.ch_headers]
        self.channel_idcs = {c['label']: idx for idx, c in enumerate(self.ch_headers)}

    def samples_to_time(self, sfreq=None):
        """Compute a time vector in seconds from sample indices."""
        if sfreq is None:
            sfreq = self.sfreq
        self.time = np.arange(self.nsamples) / sfreq

    def update_log(self):
        """Refresh cached trigger/event mappings derived from the log."""
        self.log_triggers = self.log[self.triggercol].to_numpy()
        self.trigger_mappings = dict(
            self.log[[self.triggercol, self.eventcol]].drop_duplicates().values
        )
        self.trigger_mappings_inv = dict(
            zip(self.trigger_mappings.values(), self.trigger_mappings.keys())
        )
        self.log.index = np.arange(len(self.log))

    def return_data(self, infer_codec=True):
        """Return (header, ch_headers, channels, log) ready for writing.

        If *infer_codec* is True, any trigger values missing from
        ``encodingdict`` are estimated by linear extrapolation from existing
        entries.

        .. note::
            This method modifies ``self.channels`` in-place to re-encode the
            trigger channel, and drops ``'bdfidx'`` from ``self.log`` if
            present.
        """
        if 'bdfidx' in self.log.columns:
            self.log.drop(columns='bdfidx', inplace=True)

        trigger_channel = self.trigger_channel.copy()

        missing_triggers = set(np.unique(trigger_channel)) - set(self.encodingdict)
        if infer_codec and missing_triggers:
            x = np.asarray(list(self.encodingdict.keys()))
            y = np.asarray(list(self.encodingdict.values()))
            coefs = np.polyfit(x, y, 1)
            poly = np.poly1d(coefs)
            for t in missing_triggers:
                self.encodingdict[t] = int(np.round(poly(t)))

        if self.convert_triggers:
            trigger_channel = self.encode_triggers_bdf(self.trigger_channel)

        self.channels[self.trigger_ch_idx] = trigger_channel

        return self.header, self.ch_headers, self.channels, self.log

    # ------------------------------------------------------------------ #
    # Event management                                                     #
    # ------------------------------------------------------------------ #

    def get_events(self, drop_initial_event=True, minlength=0, verbose=True):
        """Detect trigger events in the trigger channel and align with log.

        Parameters
        ----------
        drop_initial_event : bool
            Replace all occurrences of the first-sample trigger value with
            ``trigger_default`` (removes DC-offset / baseline trigger runs).
        minlength : int
            Minimum run length in samples to count as a valid event.
        verbose : bool
            Print a summary of detected events.
        """
        if drop_initial_event:
            first_val = self.trigger_channel[0]
            self.trigger_channel[self.trigger_channel == first_val] = self.trigger_default

        self.unique_triggers = np.unique(self.trigger_channel)
        self.unique_triggers = np.delete(
            self.unique_triggers,
            np.where(self.unique_triggers == self.trigger_default),
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

        if verbose:
            print(
                f'Found {len(self.bdf_triggers)} events '
                f'with unique triggers: {self.unique_triggers}.'
            )

        if len(self.log_triggers) != len(self.bdf_triggers):
            warnings.warn(
                'Uneven number of events in log and BDF file; '
                'risk of timing discrepancies. Please inspect before proceeding.',
                RuntimeWarning,
            )
        elif np.all(self.log_triggers == self.bdf_triggers):
            self.log['bdfidx'] = self.bdf_trigger_samples
        else:
            warnings.warn(
                'Log and BDF trigger codes do not match; '
                'risk of timing discrepancies. Please inspect before proceeding.',
                RuntimeWarning,
            )

    def find_event(self, lookup, mode='or', asint=True):
        """Find log row indices matching the given column/value lookup.

        Parameters
        ----------
        lookup : dict
            Mapping of ``{column_name: list_of_values}``.
        mode : {'or', 'and'}
            How to combine multi-column lookups.
        asint : bool
            Return integer index array (True) or boolean mask (False).
        """
        masks = [np.isin(self.log[col], vals) for col, vals in lookup.items()]
        combined = (
            np.logical_or.reduce(masks) if mode == 'or'
            else np.logical_and.reduce(masks)
        )
        return np.where(combined)[0] if asint else combined

    def modify_event(self, indices=None, replacements=None):
        """Delete or relabel events at the given log indices.

        Parameters
        ----------
        indices : array-like of int
            Log row indices to modify.
        replacements : dict or None
            If None, delete the events.  Otherwise a dict mapping
            ``column_name → new_values`` applied to those rows.
        """
        if len(indices) == 0:
            return

        bdfidcs = self.log.loc[indices, 'bdfidx'].to_list()
        idcsflat = np.concatenate(bdfidcs)

        if replacements is None:
            self.log.drop(indices, inplace=True)
            self.log.index = np.arange(len(self.log))
            self.trigger_channel[idcsflat] = self.trigger_default
        else:
            for key, val in replacements.items():
                self.log.loc[indices, key] = pd.Series(val, index=indices)
                if key == self.triggercol:
                    repeat_counts = np.array([len(lst) for lst in bdfidcs])
                    if isinstance(val, (np.ndarray, list)):
                        val = np.repeat(val, repeat_counts)
                    self.trigger_channel[idcsflat] = val

        self.update_log()
        self.get_events(verbose=False)

    def insert_event(self, indices, events, trigger_span=15, allow_overlap=False):
        """Insert new events into the log and trigger channel.

        Parameters
        ----------
        indices : array-like of int
            Log positions at which to insert.
        events : dict
            Event definition with keys matching log columns; must include
            either ``'bdfidx'`` or ``'onset'`` to locate the trigger.
        trigger_span : int
            Trigger pulse width in samples.
        allow_overlap : bool
            Whether to allow inserting over existing triggers.
        """
        if 'bdfidx' in events:
            self.insert_triggers_bdf(
                events[self.triggercol],
                samples=events['bdfidx'],
                trigger_span=trigger_span,
                allow_overlap=allow_overlap,
            )
        elif 'onset' in events:
            self.insert_triggers_bdf(
                events[self.triggercol],
                times=events['onset'],
                trigger_span=trigger_span,
                allow_overlap=allow_overlap,
            )

        self.insert_event_log(indices, events)
        self.get_events(verbose=False)

    def insert_event_log(self, indices, events):
        """Insert rows into the event log at the specified positions.

        Parameters
        ----------
        indices : array-like of int
            Positions in the current log at which to insert rows.
        events : dict
            Column → value mapping for the new rows.  Missing columns are
            filled with NaN.
        """
        if isinstance(events, pd.DataFrame):
            events = dict(events)

        insert = {
            col: (events[col] if col in events else np.full(len(indices), np.nan))
            for col in self.log.columns
        }

        insert_arr = pd.DataFrame(insert).to_numpy()
        lognp = self.log.to_numpy()
        self.log = pd.DataFrame(
            np.insert(lognp, indices, insert_arr, axis=0),
            columns=self.log.columns,
        )
        self.update_log()

    def insert_triggers_bdf(self, triggers, times=None, samples=None,
                            trigger_span=15, allow_overlap=False):
        """Write trigger codes into the trigger channel.

        Parameters
        ----------
        triggers : array-like
            Trigger code(s) to insert.
        times : array-like of float, optional
            Onset times in seconds (converted to samples via ``sfreq``).
        samples : array-like of int or list of arrays, optional
            Sample indices.  Scalars are expanded to ``trigger_span``-length
            pulses; pre-built index arrays are used as-is.
        trigger_span : int
            Pulse width in samples when *samples* contains scalar start points.
        allow_overlap : bool
            If False, raise an error when existing triggers would be overwritten.
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
            np.isin(self.trigger_channel[samples_flat], self.unique_triggers)
        ):
            raise ValueError('Triggers already exist at the requested time points.')

        self.trigger_channel[samples_flat] = triggers_flat

    def insert_span(self, start, stop=None, length=None, value=0):
        """Insert a silent span (constant value) into channels and trigger.

        Parameters
        ----------
        start : float
            Insertion point in seconds.
        stop : float, optional
            End of the span in seconds (used to derive *length* if not given).
        length : int, optional
            Span length in samples.  Derived from ``stop - start`` if None.
        value : numeric
            Fill value for the new span.
        """
        if length is None:
            length = int((stop - start) * self.sfreq)

        start_sample = int(start * self.sfreq)

        span = np.full((self.channels.shape[0], length), value, dtype=self.dtype)
        self.channels = np.concatenate(
            [self.channels[:, :start_sample], span, self.channels[:, start_sample:]],
            axis=1,
        )

        trigspan = np.full(length, value)
        self.trigger_channel = np.concatenate([
            self.trigger_channel[:start_sample],
            trigspan,
            self.trigger_channel[start_sample:],
        ])

        self.get_events(verbose=False)
        self.update_log()

    # ------------------------------------------------------------------ #
    # Channel management                                                   #
    # ------------------------------------------------------------------ #

    def drop_channels(self, channels):
        """Remove channels by name.

        Parameters
        ----------
        channels : list of str
            Channel labels to remove.
        """
        ch_idcs = [self.channel_idcs[ch] for ch in channels]
        self.channels = np.delete(self.channels, ch_idcs, axis=0)
        self.ch_headers = [
            ch for idx, ch in enumerate(self.ch_headers) if idx not in ch_idcs
        ]
        self.get_channels()

    def crop_events(self, start, stop, column, inplace=False):
        """Crop the session to the time window surrounding events in *column*.

        Parameters
        ----------
        start : numeric
            Lower bound of the *column* range to keep.
        stop : numeric
            Upper bound of the *column* range to keep.
        column : str
            Log column to filter on.
        inplace : bool
            If True, modify self in-place; if False, return a new
            SessionHandler.
        """
        query = self.log[column].to_numpy()
        indices = np.where(np.logical_and(query >= start, query <= stop))[0]
        cropped_log = self.log.loc[indices].copy()

        start_diff = cropped_log[column].to_numpy()[0] - start
        stop_diff = stop - cropped_log[column].to_numpy()[-1]

        bdfspan = np.concatenate(cropped_log['bdfidx'].to_list())
        bdfstart = bdfspan[0] - int(start_diff * self.sfreq)
        bdfstop = bdfspan[-1] + int(stop_diff * self.sfreq)

        copytrigger = self.trigger_channel.copy()
        copytrigger[:bdfspan[0]] = self.trigger_default
        copytrigger[bdfspan[-1] + 1:] = self.trigger_default

        cropped_channels = self.channels[:, bdfstart:bdfstop].copy()
        cropped_triggers = copytrigger[bdfstart:bdfstop]

        if inplace:
            self.channels = cropped_channels
            self.log = cropped_log
            self.update_log()
            self.get_events()
        else:
            if self.convert_triggers:
                cropped_triggers = self.encode_triggers_bdf(cropped_triggers)

            cropped_channels[self.trigger_ch_idx] = cropped_triggers
            cropped_log = cropped_log.drop(columns='bdfidx')

            return self.__class__(
                cropped_log,
                cropped_channels,
                self.ch_headers,
                self.header,
                trigger_ch_name=self.trigger_ch_name,
                convert_triggers=self.convert_triggers,
                default_trigger=self.trigger_default,
                eventcol=self.eventcol,
                triggercol=self.triggercol,
                drop_initial_event=False,
            )

    # ------------------------------------------------------------------ #
    # Trigger encoding                                                     #
    # ------------------------------------------------------------------ #

    def encode_triggers_bdf(self, triggers):
        """Re-encode trigger values into the original BDF integer format.

        Upper 16 bits come from ``encodingdict``; lower 16 bits carry the
        decoded trigger code.
        """
        encoded = np.asarray([self.encodingdict[t] for t in triggers])
        result = (
            encoded.astype(np.uint32) & 0xFFFF0000
            | triggers.astype(np.uint32) & 0x0000FFFF
        )
        return result.astype(np.int32)

    # ------------------------------------------------------------------ #
    # Repair methods (originally repair_funcs.py)                         #
    # ------------------------------------------------------------------ #

    def repair_aberrant_trigger(self, maxlen):
        """Replace aberrant triggers shorter than *maxlen* samples.

        Aberrant pulses are replaced by their neighbouring trigger code when
        adjacent, or by ``trigger_default`` otherwise.

        Parameters
        ----------
        maxlen : int
            Minimum trigger pulse length in samples; shorter pulses are
            considered aberrant.
        """
        n = len(self.bdf_trigger_samples)
        for idx, t in enumerate(self.bdf_trigger_samples):
            if len(t) < maxlen:
                if idx > 0 and np.any(np.isin(t - 1, self.bdf_trigger_samples[idx - 1])):
                    replacement = self.bdf_triggers[idx - 1]
                elif idx < n - 1 and np.any(
                    np.isin(t + 1, self.bdf_trigger_samples[idx + 1])
                ):
                    replacement = self.bdf_triggers[idx + 1]
                else:
                    replacement = self.trigger_default
                self.trigger_channel[t] = replacement
        self.get_events()

    def fix_false_starts(self):
        """Trim false-start events from the log and trigger channel.

        Finds the last occurrence of trigger code 1 in the log and discards
        everything before it, zeroing the corresponding trigger channel samples.
        """
        starts_log = self.find_event({self.triggercol: [1]})
        self.log = self.log.iloc[starts_log[-1]:]
        self.update_log()

        starts_bdf = np.where(self.bdf_triggers == 1)[0]
        if len(starts_bdf) > 1:
            sample = self.bdf_trigger_samples[starts_bdf[-1] - 1][-1]
            self.trigger_channel[:sample + 1] = 0

        self.get_events()

    def insert_missing_bdf(self):
        """Interpolate and insert BDF triggers that are in the log but absent
        from the recording, using log timing as a reference.

        Raises
        ------
        AssertionError
            If the last log trigger does not match the last BDF trigger.
        RuntimeError
            If the algorithm fails to converge within the expected number of
            iterations.
        """
        assert (
            self.log[self.triggercol].to_numpy()[-1] == self.bdf_triggers[-1]
        ), 'Last log trigger does not match last BDF trigger.'

        loglen = self.log.iloc[-1]['time'] - self.log.iloc[0]['time']
        bdflen = self.trigger_times[-1] - self.trigger_times[0]

        if loglen > bdflen:
            self.insert_span(start=0, stop=(loglen - bdflen) + 100)
            self.get_events()

        trig_diff = len(self.log) - len(self.bdf_triggers)
        print(f'Detected {trig_diff} missing triggers.')
        counter = tqdm(range(trig_diff), desc='Interpolating triggers')

        attempts = 0
        while len(self.log) != len(self.bdf_triggers):
            if attempts > len(self.log) + 1000:
                raise RuntimeError(
                    'Maximum attempts exceeded while inserting missing triggers.'
                )

            for idx, row in self.log.iterrows():
                if self.bdf_triggers[idx] != row[self.triggercol]:
                    tdiff = self.log.iloc[-1]['time'] - row['time']
                    samples = self.bdf_trigger_samples[-1] - int(self.sfreq * tdiff)
                    self.insert_triggers_bdf(
                        triggers=[row[self.triggercol]], samples=[samples]
                    )
                    attempts = 0
                    self.update_log()
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        self.get_events(verbose=False)
                    counter.update()
                    break

            attempts += 1

        self.update_log()
        self.get_events()
