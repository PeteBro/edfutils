"""EEG recording management object with EDF/BDF I/O via pyedflib."""

import os
import copy
import warnings

import numpy as np
import pandas as pd
import pyedflib
from tqdm import tqdm

class RecordingHandler:
    """Manages EEG recordings in EDF/BDF format alongside a behavioural log.

    Loads signal data via pyedflib and a CSV event log, aligns trigger codes
    between the two, and supports cropping, insertion, and re-export.

    Attributes:
        eeg_path (str): Absolute path to the EDF/BDF file.
        log_path (str): Absolute path to the CSV log file.
        log (pd.DataFrame): Behavioural event log.
        trigger_column (str): Log column name containing trigger/event codes.
        onset_column (str): Log column name containing event onset times.
        sfreq (int): Sampling frequency in Hz.
        dtype (np.dtype): Data type used for signal arrays.
        filetype: pyedflib file type constant.
        digital (bool): Whether signals are read as digital values.
        bytemask (int): Bitmask used to decode/encode the trigger channel.
        encodingdict (dict): Mapping of raw to decoded trigger values.
    """

    def __init__(self, eeg_path, log_path, trigger_channel_name='Status', trigger_column='trigger', onset_column='onset'):
        """
        Args:
            eeg_path (str): Path to the EDF/BDF recording file.
            log_path (str): Path to the CSV behavioural log.
            trigger_column (str): Log column containing trigger/event codes
                (must be integer dtype).
            onset_column (str): Log column containing event onset times in
                seconds (must be float dtype).
        """
        self.log_path = os.path.abspath(log_path)
        self.eeg_path = os.path.abspath(eeg_path)

        self.read_log(self.log_path, trigger_column=trigger_column, onset_column=onset_column)

        self.reader = pyedflib.EdfReader(self.eeg_path)

        self._channels_loaded = []
        self._data = []

        ######################################
        self.dtype = np.int32
        self.sfreq=2048
        self.filetype = self.reader.filetype
        self.digital = True
        self._trigger_channel_name = trigger_channel_name
        ######################################

        self.encodingdict = {}

        self.read_data(channels = [self._trigger_channel_name])
        self.bytemask = self.get_bytemask(self._trigger_channel_name)

        if self.filetype == 'BDF':
            self._data[self._trigger_idx] = self.decode_channels(self.trigger_channel)

    ############
    # Properties
    ############

    @property
    def nsamples(self):
        """int: Total number of samples in the recording (from channel 0)."""
        return self.reader.samples_in_file(0)

    @property
    def channels(self):
        """list[str]: All signal labels present in the EDF/BDF file."""
        return self.reader.getSignalLabels()

    @property
    def datarecord_duration(self):
        """float: Duration of one data record in seconds."""
        return self.reader.datarecord_duration

    @property
    def trigger_channel(self):
        """np.ndarray: The loaded trigger/status channel samples."""
        return self._data[self._trigger_idx]

    @property
    def n_channels(self):
        """int: Number of channels currently loaded into ``_data``."""
        return len(self._data)

    @property
    def trigger_default(self):
        """int: Most common value in the trigger channel (baseline/idle state)."""
        return np.bincount(self.trigger_channel).argmax()

    @property
    def ch_headers(self):
        """list[dict]: Signal headers for each loaded channel."""
        return [self.reader.getSignalHeader(ch) for ch in self._channels_loaded]

    @property
    def header(self):
        """dict: File-level header from the EDF/BDF reader."""
        return self.reader.getHeader()

    ##########################
    # Simple reading functions
    ##########################

    def get_bytemask(self, channel):
        """Compute a bitmask that isolates the trigger bits in a status channel.

        Derives the mask from the channel's digital range and its most common
        (baseline) value, so that ``data & bytemask`` strips the baseline bits.

        Args:
            channel (str): Name of the channel to compute the mask for.

        Returns:
            int: Bitmask with baseline bits cleared.
        """
        ch_idx = self._channels_loaded.index(channel)
        header = self.ch_headers[ch_idx]
        n_bits = int(np.log2(header['digital_max'] - header['digital_min'] + 1))
        bit_cap = (1 << n_bits) - 1
        ch_data = self._data[ch_idx]
        baseline = np.bincount(ch_data).argmax()
        return ~baseline & bit_cap


    def decode_channels(self, data):
        """Apply the bytemask to strip baseline bits from trigger data.

        Also updates ``encodingdict`` with the raw-to-decoded mapping.

        Args:
            data (np.ndarray): Raw digital trigger channel samples.

        Returns:
            np.ndarray: Decoded trigger samples.
        """
        decoded_data = data & self.bytemask
        self.encodingdict.update(dict(zip(data, decoded_data)))

        return decoded_data


    def read_data(self, channels=None):
        """Read signal data from the EDF/BDF file into ``_data``.

        Skips channels already loaded. After loading, re-sorts ``_data`` and
        ``_channels_loaded`` to match the original file order.

        .. note::
            ``_data`` is replaced on each call (via ``vstack``), invalidating
            any previously cached views — always access ``trigger_channel``
            through the property, never store it in a variable across calls.

        Args:
            channels (list[str], optional): Channel names to load. Defaults to
                all channels in the file.
        """
        if channels is None:
            channels = self.channels
        ch_idcs = [self.channels.index(ch) for ch in channels]
        for idx, ch in tqdm(zip(ch_idcs, channels), total=len(channels)):
            
            if ch in self._channels_loaded:
                continue
            else:
                self._channels_loaded.append(ch)
                new_row = self.reader.readSignal(idx, digital=self.digital).astype(self.dtype)
            
            if len(self._data) == 0:
                self._data = new_row[None, :]
            else:
                self._data = np.vstack([self._data, new_row])

        order = np.argsort([self.channels.index(ch) for ch in self._channels_loaded])
        self._data = self._data[order]
        self._channels_loaded = [self._channels_loaded[idx] for idx in order]
        self._trigger_idx = self._channels_loaded.index(self._trigger_channel_name)


    def read_log(self, log_path, trigger_column, onset_column):
        """Load the behavioural log into ``self.log`` and validate key columns.

        The delimiter is detected automatically. Warns if a column name is not
        provided or has an unexpected dtype; raises if the column is not found.

        Args:
            log_path (str): Path to the log file.
            trigger_column (str): Column name for trigger/event codes.
            onset_column (str): Column name for event onset times.

        Raises:
            ValueError: If either column is not present in the log.
        """
        self.log = pd.read_csv(log_path, sep=None, engine='python')

        cols = [('trigger_column', trigger_column, pd.api.types.is_integer_dtype, 'int'),
                ('onset_column', onset_column, pd.api.types.is_float_dtype, 'float')]

        for attr, colname, typecheck, expected in cols:

            if colname not in self.log.columns:
                    
                raise ValueError(
                    f"Column '{colname}' not found in log. "
                    f"Available columns: {list(self.log.columns)}"
                )
            
            elif not typecheck(self.log[colname]):
                
                warnings.warn(
                    f"Column '{colname}' has dtype {self.log[colname].dtype}, expected {expected}.",
                    UserWarning,
                )
            
            setattr(self, attr, colname)


    def update_log(self):
        """Refresh log-derived attributes from the current state of ``self.log``.

        Updates ``log_triggers``, ``log_times``, ``unique_log_triggers``, and
        ``trigger_events``, and resets the log index.
        """
        self.log_triggers = self.log[self.trigger_column].to_numpy()
        self.log_times = self.log[self.onset_column].to_numpy()
        self.unique_log_triggers = np.unique(self.log_triggers)

        self.log = self.log.reset_index(drop=True)


    def get_eeg_events(self, drop_initial_event=True, minlength=0, verbose=True):
        """Extract trigger events from the EEG trigger channel and align with the log.

        Populates ``bdf_trigger_samples``, ``bdf_triggers``, ``trigger_times``,
        and ``unique_eeg_triggers``. If the log and EEG triggers match,
        ``file_idcs`` is added to ``self.log``. Warns on count or code mismatches.

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
                f'Found {len(self.bdf_triggers)} events in data file'
                f'with unique triggers: {self.unique_eeg_triggers}.'
            )

        if len(self.log_triggers) != len(self.bdf_triggers):
            warnings.warn(
                'Uneven number of events in log and BDF file; '
                'risk of timing discrepancies. Please run diagnostics before proceeding.',
                RuntimeWarning,
            )

        elif not np.all(self.log_triggers == self.bdf_triggers):
            warnings.warn(
                'Log and BDF trigger codes do not match; '
                'risk of timing discrepancies. Please run diagnostics before proceeding.',
                RuntimeWarning,
            )

        else:
            self.log['file_idcs'] = self.bdf_trigger_samples

    #############################################
    # Data exploration and modification functions
    #############################################

    def find_event(self, lookup, mode='or', asint=True):
        """Find log row indices matching a set of column-value filters.

        Args:
            lookup (dict): Mapping of column name to value or list of values.
            mode (str): ``'or'`` to match any filter, ``'and'`` to match all.
            asint (bool): If True, return integer indices; otherwise return a
                boolean mask.

        Returns:
            np.ndarray: Matching row indices or boolean mask.
        """
        mask = pd.DataFrame(np.full(self.log.shape, True),
                            columns = self.log.columns)
        
        for key, vals in lookup.items():
            if not(isinstance(vals, (list, tuple))):
                vals = [vals]
            mask[key] &= self.log[key].isin(vals)

        mask = mask.to_numpy()
        if mode == 'or':
            mask = np.any(mask, axis=1)
        elif mode == 'and':
            mask = np.all(mask, axis=1)

        return np.where(mask)[0] if asint else mask


    def insert_log_event(self, indices, events):
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

        insert_arr = pd.DataFrame(events).to_numpy()
        lognp = self.log.to_numpy()
        self.log = pd.DataFrame(
            np.insert(lognp, indices, insert_arr, axis=0),
            columns=self.log.columns,
        )
        self.update_log()


    def insert_eeg_triggers(self, triggers, times=None, samples=None,
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


    def insert_event(self, indices, events, trigger_span=15, allow_overlap=False):
        """Insert an event into both the log and the EEG trigger channel.

        Uses ``file_idcs`` if present in ``events``, otherwise falls back to
        ``onset_column``. Refreshes events via ``get_eeg_events`` afterwards.

        Args:
            indices (int or array-like): Log row position(s) for insertion.
            events (dict or pd.DataFrame): Event data; must include trigger and
                timing information.
            trigger_span (int): Passed to ``insert_eeg_triggers``.
            allow_overlap (bool): Passed to ``insert_eeg_triggers``.
        """
        if 'file_idcs' in events:
            self.insert_eeg_triggers(
                events[self.trigger_column],
                samples=events['file_idcs'],
                trigger_span=trigger_span,
                allow_overlap=allow_overlap,
            )

        elif self.onset_column in events:
            self.insert_eeg_triggers(
                events[self.trigger_column],
                times=events[self.onset_column],
                trigger_span=trigger_span,
                allow_overlap=allow_overlap,
            )

        self.insert_log_event(indices, events)
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
        ch_idcs = [self._channels_loaded.index(ch) for ch in channels]
        self._data = np.delete(self._data, ch_idcs, axis=0)
        self._channels_loaded = np.delete(self._channels_loaded, ch_idcs).tolist()


    def crop(self, start, stop, column, inplace=False):
        """Crop the recording to a range of values in a log column.

        Selects log rows where ``column`` is between ``start`` and ``stop``,
        then slices ``_data`` to the corresponding sample range (with padding
        to cover the full requested interval).

        Args:
            start: Lower bound of the column range (inclusive).
            stop: Upper bound of the column range (inclusive).
            column (str): Log column to filter on.
            inplace (bool): If True, modify this object; otherwise return a
                deep copy with the cropped data.

        Returns:
            RecordingHandler or None: Cropped copy if ``inplace=False``.
        """
        query = self.log[column].to_numpy()
        indices = np.where(np.logical_and(query >= start, query <= stop))[0]
        cropped_log = self.log.loc[indices].copy()

        start_diff = cropped_log[column].to_numpy()[0] - start
        stop_diff = stop - cropped_log[column].to_numpy()[-1]

        bdfspan = np.concatenate(cropped_log['file_idcs'].to_list())
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
        
    #################################
    # Diagnostic and repair functions
    #################################

    def run_diagnostic(self):

        pass

    ########################
    # Data writing functions
    ########################


    def encode_channels(self, data):
        """Re-encode decoded trigger data by restoring the baseline bits.

        Inverse of ``decode_channels``: ORs the data with the inverted bytemask
        to reinstate the bits that were stripped on read.

        Args:
            data (np.ndarray): Decoded trigger channel samples.

        Returns:
            np.ndarray: Re-encoded trigger samples ready for writing.
        """
        return (data | (~self.bytemask & 0xFFFFFF)).astype(self.dtype)

    
    def fill_datarecords(self):
        """Pad ``_data`` with zeros so its length is a multiple of the data record size.

        Required before writing, as EDF/BDF files must contain whole records.
        """
        remainder = self._data.shape[-1] % (self.datarecord_duration*self.sfreq)
        if remainder:
            filler = np.zeros((self._data.shape[0], int(self.sfreq-remainder))).astype(self.dtype)
            self._data = np.concatenate([self._data, filler], axis=1)
    

    def write_log(self, path, **kwargs):
        """Write the event log to a CSV file.

        Args:
            path (str): Destination file path.
            **kwargs: Additional arguments passed to ``pd.DataFrame.to_csv``.
        """
        self.log.to_csv(path, **kwargs)


    def write_data(self, path, verbose=True):
        """Write the loaded signal data to an EDF/BDF file.

        Re-encodes the trigger channel for BDF files before writing. Data is
        written in whole data records via ``blockWriteDigitalSamples``.

        Args:
            path (str): Destination file path. May be the same as ``eeg_path``
                for in-place writes.
            verbose (bool): If True, show a tqdm progress bar.
        """
        if self.filetype == 'BDF':
            self._data[self._trigger_idx] = self.encode_channels(self.trigger_channel)

        writer = pyedflib.EdfWriter(path, self.n_channels, file_type=self.filetype)
        
        writer.setHeader(self.header)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            writer.setSignalHeaders(self.ch_headers)
        
        ndatarecords = self._data.shape[-1] // int(self.datarecord_duration*self.sfreq)
        data = np.split(self._data, ndatarecords, axis=1)
        for record in tqdm(data, desc='Writing data records', disable=not verbose):
            _ = writer.blockWriteDigitalSamples(record.flatten())        
        writer.close()

