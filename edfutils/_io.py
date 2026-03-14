"""I/O mixin for EEGSession — reading and writing EDF/BDF and log files."""

import warnings

import numpy as np
import pandas as pd
import pyedflib
from tqdm import tqdm


class _IOMixin:

    def read_data(self, channels=None, verbose=True):
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

        for ch in tqdm(channels, total=len(channels), disable=not verbose):
            if self.channel.loaded(ch):
                continue
            new_row = self.channel.load(ch)
            self._channels_loaded.append(ch)
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


    def fill_datarecords(self):
        """Pad ``_data`` with zeros so its length is a multiple of the data record size.

        Required before writing, as EDF/BDF files must contain whole records.
        """
        record_samples = int(self.datarecord_duration * self.sfreq)
        remainder = self._data.shape[-1] % record_samples
        if remainder:
            filler = np.zeros((self._data.shape[0], record_samples - remainder)).astype(self.dtype)
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
            self._data[self._trigger_idx] = self.channel.encode(self._trigger_channel_name)

        writer = pyedflib.EdfWriter(path, self.n_channels, file_type=self.filetype)

        writer.setHeader(self.header)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            writer.setSignalHeaders(self.ch_headers)

        ndatarecords = self._data.shape[-1] // int(self.datarecord_duration * self.sfreq)
        data = np.split(self._data, ndatarecords, axis=1)
        for record in tqdm(data, desc='Writing data records', disable=not verbose):
            _ = writer.blockWriteDigitalSamples(record.flatten())
        writer.close()
