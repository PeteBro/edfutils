"""Diagnostics and repair mixin for EEGSession."""

import warnings

import numpy as np
import pandas as pd


class _DiagnosticsMixin:
    """Diagnostics and targeted repairs for EEG/log alignment issues."""

    def _check_event_count(self):
        raise NotImplementedError


    def _check_trigger_mismatch(self):

        raise NotImplementedError


    def _check_time_aligned(self):

        raise NotImplementedError


    def _check_false_starts(self, start_trigger=1):

        raise NotImplementedError


    def _check_trigger_pulse_lengths(self, minlen=None, maxlen=None):

        raise NotImplementedError


    def repair_aberrant_trigger(self, minlen=None, maxlen=None):

        raise NotImplementedError
        

    def rebuild_from_log(self):

        raise NotImplementedError