"""Diagnostics and repair mixin for EEGSession."""


class _DiagnosticsMixin:

    def run_diagnostic(self):
        """Run diagnostics on the session to identify data quality issues.

        .. note::
            Not yet implemented.
        """
        pass

    def repair_aberrant_trigger(self, maxlen):
        """Remove spurious trigger pulses exceeding ``maxlen`` samples.

        .. note::
            Not yet implemented.

        Args:
            maxlen (int): Maximum allowed trigger pulse length in samples.
        """
        pass

    def fix_false_starts(self):
        """Remove events recorded before the experiment actually began.

        .. note::
            Not yet implemented.
        """
        pass

    def merge_recordings(self, other):
        """Merge another EEGSession into this one.

        .. note::
            Not yet implemented.

        Args:
            other (EEGSession): The session to merge in.
        """
        pass

    def insert_missing_bdf(self):
        """Insert EEG trigger codes present in the log but absent from the BDF.

        .. note::
            Not yet implemented.
        """
        pass
