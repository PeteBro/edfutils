"""Per-channel accessor for EEGSession."""

import numpy as np


class _ChannelAccessor:
    """Namespace for per-channel lookups on an EEGSession."""

    def __init__(self, session):
        self._session = session

    def index(self, name):
        """Return the row index of ``name`` in ``_data``."""
        return self._session._channels_loaded.index(name)

    def nbits(self, name):
        """Return the bit depth of channel ``name``."""
        header = self.header(name)
        return int(np.log2(header['digital_max'] - header['digital_min'] + 1))

    def bytemask(self, name):
        """Return the bitmask used to decode channel ``name``."""
        nbits = self.nbits(name)
        bitcap = (1 << nbits) - 1
        idx = self.index(name)
        values, counts = np.unique(self._session._data[idx], return_counts=True)
        baseline = values[counts.argmax()]
        return ~baseline & bitcap

    def sfreq(self, name):
        """Return the sampling frequency of channel ``name`` in Hz."""
        idx = self._file_index(name)
        return self._session.reader.getSampleFrequency(idx)

    def dtype(self, name):
        """Return the numpy dtype for channel ``name``."""
        return np.int32 if self.nbits(name) == 24 else np.int16

    def header(self, name):
        """Return the signal header dict for channel ``name``."""
        idx = self._file_index(name)
        return self._session.reader.getSignalHeader(idx)

    def loaded(self, name):
        """Return whether channel ``name`` is currently loaded into ``_data``."""
        return name in self._session._channels_loaded

    def load(self, name):
        """Read channel ``name`` from the EDF/BDF file and return its samples.

        Args:
            name (str): Channel name to read.

        Returns:
            np.ndarray: Raw digital samples for the channel.
        """
        idx = self._file_index(name)
        return self._session.reader.readSignal(idx, digital=self._session.digital).astype(self.dtype(name))

    def decode(self, name):
        """Return a decoded copy of channel ``name`` with baseline bits stripped.

        Args:
            name (str): Channel name.

        Returns:
            np.ndarray: Decoded channel samples.
        """
        idx = self.index(name)
        return self._session._data[idx].copy() & self.bytemask(name)

    def encode(self, name):
        """Return a re-encoded copy of channel ``name`` with baseline bits restored.

        Inverse of :meth:`decode`.

        Args:
            name (str): Channel name.

        Returns:
            np.ndarray: Re-encoded channel samples.
        """
        idx = self.index(name)
        nbits = self.nbits(name)
        mask = self.bytemask(name)
        return (self._session._data[idx] | (~mask & ((1 << nbits) - 1))).astype(self.dtype(name))

    def drop(self, name):
        """Remove channel ``name`` from ``_data`` and ``_channels_loaded``.

        Args:
            name (str): Channel name to drop.
        """
        idx = self.index(name)
        self._session._data = np.delete(self._session._data, idx, axis=0)
        self._session._channels_loaded.remove(name)
        self._session._trigger_idx = self._session._channels_loaded.index(
            self._session._trigger_channel_name
        )

    def _file_index(self, name):
        """Return the file-level index of channel ``name`` as stored in the EDF/BDF."""
        return self._session.channels.index(name)
