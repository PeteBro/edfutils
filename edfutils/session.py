"""EEG recording session — core class combining all mixins."""

import os

import numpy as np
import pyedflib

from ._channel import _ChannelAccessor
from ._io import _IOMixin
from ._editing import _EditingMixin
# from ._diagnostics import _DiagnosticsMixin  # not yet implemented


class EEGSession(_IOMixin, _EditingMixin):
    """Manages EEG recordings in EDF/BDF format alongside a behavioural log.

    Loads signal data via pyedflib and a CSV event log, aligns trigger codes
    between the two, and supports cropping, insertion, and re-export.

    .. note::
        Only digital signal reading (``digital=True``) is currently supported.
        Passing ``digital=False`` is accepted but untested and may produce
        incorrect results.

    Attributes:
        eeg_path (str): Absolute path to the EDF/BDF file.
        log_path (str): Absolute path to the CSV log file.
        log (pd.DataFrame): Behavioural event log.
        trigger_column (str): Log column name containing trigger/event codes.
        onset_column (str): Log column name containing event onset times.
        channel (_ChannelAccessor): Per-channel lookups (index, nbits, bytemask, sfreq, dtype).
    """

    def __init__(self, eeg_path, log_path, trigger_channel_name='Status',
                 trigger_column='trigger', onset_column='onset', digital=True):
        """
        Args:
            eeg_path (str): Path to the EDF/BDF recording file.
            log_path (str): Path to the CSV behavioural log.
            trigger_channel_name (str): Label of the trigger/status channel in
                the EDF/BDF file. Defaults to ``'Status'``.
            trigger_column (str): Log column containing trigger/event codes
                (must be integer dtype).
            onset_column (str): Log column containing event onset times in
                seconds (must be float dtype).
            digital (bool): Whether to read signals as digital values. Only
                ``True`` is currently supported.
        """
        self.log_path = os.path.abspath(log_path)
        self.eeg_path = os.path.abspath(eeg_path)

        self.read_log(self.log_path, trigger_column=trigger_column, onset_column=onset_column)

        self.reader = pyedflib.EdfReader(self.eeg_path)

        self._channels_loaded = []
        self._data = []

        self.digital = digital
        self._trigger_channel_name = trigger_channel_name

        self.read_data(channels=[self._trigger_channel_name], verbose=False)
        self._data[self._trigger_idx] = self.channel.decode(self._trigger_channel_name)

    ############
    # Properties
    ############

    @property
    def nsamples(self):
        """int: Total number of samples in the recording (from channel 0)."""
        return self._data.shape[-1]

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
        values, counts = np.unique(self.trigger_channel, return_counts=True)
        return values[counts.argmax()]

    @property
    def ch_headers(self):
        """list[dict]: Signal headers for each loaded channel."""
        return [self.channel.header(ch) for ch in self._channels_loaded]

    @property
    def header(self):
        """dict: File-level header from the EDF/BDF reader."""
        return self.reader.getHeader()

    @property
    def filetype(self):
        """int: pyedflib file type constant (e.g. EDF, BDF)."""
        return self.reader.filetype

    @property
    def sfreq(self):
        """float: Sampling frequency in Hz, asserted uniform across loaded channels.

        Raises:
            ValueError: If loaded channels have different sampling frequencies.
        """
        freqs = np.unique([self.channel.sfreq(ch) for ch in self._channels_loaded])
        if len(freqs) > 1:
            raise ValueError(
                f"{len(freqs)} different sample frequencies detected across loaded channels, must be uniform."
            )
        return freqs[0]

    @property
    def channel(self):
        """_ChannelAccessor: Per-channel lookups (index, nbits, bytemask, sfreq, dtype)."""
        return _ChannelAccessor(self)

    @property
    def dtype(self):
        """np.dtype: Widest numpy dtype required across loaded channels (int32 if any channel is 24-bit, else int16)."""
        return np.int32 if any(self.channel.nbits(ch) == 24 for ch in self._channels_loaded) else np.int16
