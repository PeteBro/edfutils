"""Diagnostics and repair mixin for EEGSession."""

import warnings

import numpy as np
import pandas as pd


class _DiagnosticsMixin:
    """Diagnostics and targeted repairs for EEG/log alignment issues."""

    _diagnostic_columns = [
        'issue',
        'severity',
        'log_index',
        'eeg_index',
        'trigger',
        'sample',
        'message',
        'repair',
    ]

    def run_diagnostic(
        self,
        start_trigger=1,
        min_trigger_len=None,
        max_trigger_len=None,
        check_false_starts=True,
        check_pulse_lengths=True,
        check_alignment=True,
    ):
        """Run non-mutating diagnostics on log and EEG trigger alignment.

        Args:
            start_trigger (int): Trigger code marking the real experiment start.
            min_trigger_len (int, optional): Report trigger pulses shorter than
                this many samples.
            max_trigger_len (int, optional): Report trigger pulses longer than
                this many samples.
            check_false_starts (bool): If True, report repeated start triggers.
            check_pulse_lengths (bool): If True, report suspicious pulse lengths.
            check_alignment (bool): If True, compare log and EEG trigger codes.

        Returns:
            pd.DataFrame: One row per issue. Also stored as
            ``self.diagnostics``.
        """
        self._refresh_diagnostic_events()

        issues = []
        
        issues.extend(self._check_event_count())
        issues.extend(self._check_trigger_mismatches())
        issues.extend(self._check_false_starts(start_trigger=start_trigger))
        issues.extend(self._check_trigger_pulse_lengths(minlen=min_trigger_len, maxlen=max_trigger_len))

        self.diagnostics = pd.DataFrame(issues, columns=self._diagnostic_columns)
        return self.diagnostics

    def _refresh_diagnostic_events(self):
        """Refresh event attributes while suppressing expected mismatch warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.get_eeg_events(verbose=False)

    def _issue(
        self,
        issue,
        severity,
        message,
        log_index=None,
        eeg_index=None,
        trigger=None,
        sample=None,
        repair=None,
    ):
        """Build one diagnostic issue record."""
        return {
            'issue': issue,
            'severity': severity,
            'log_index': log_index,
            'eeg_index': eeg_index,
            'trigger': trigger,
            'sample': sample,
            'message': message,
            'repair': repair,
        }

    def _check_event_count(self):
        """Report unequal numbers of log and EEG events."""
        n_log = len(self.log_triggers)
        n_eeg = len(self.eeg_triggers)
        if n_log == n_eeg:
            return []

        if n_log > n_eeg:
            issue = 'missing_eeg_triggers'
            repair = 'insert_missing_eeg_triggers'
            message = f'Log has {n_log - n_eeg} more events than EEG triggers.'
        else:
            issue = 'extra_eeg_triggers'
            repair = 'inspect_or_modify_event'
            message = f'EEG has {n_eeg - n_log} more triggers than log events.'

        return [
            self._issue(
                issue=issue,
                severity='error',
                message=message,
                repair=repair,
            )
        ]

    def _check_trigger_mismatches(self):
        """Report aligned positions where log and EEG trigger codes differ."""
        n_compare = min(len(self.log_triggers), len(self.eeg_triggers))
        if n_compare == 0:
            return []

        log_triggers = self.log_triggers[:n_compare]
        eeg_triggers = self.eeg_triggers[:n_compare]
        mismatch_idcs = np.where(log_triggers != eeg_triggers)[0]

        issues = []
        for idx in mismatch_idcs:
            sample = self.eeg_trigger_samples[idx][0]
            issues.append(
                self._issue(
                    issue='trigger_mismatch',
                    severity='error',
                    log_index=int(idx),
                    eeg_index=int(idx),
                    trigger=int(eeg_triggers[idx]),
                    sample=int(sample),
                    message=(
                        f'Log trigger {log_triggers[idx]} does not match '
                        f'EEG trigger {eeg_triggers[idx]} at aligned event {idx}.'
                    ),
                    repair='inspect_or_modify_event',
                )
            )

        return issues

    def _check_false_starts(self, start_trigger=1):
        """Report repeated start triggers in the log or EEG data."""
        issues = []

        starts_log = np.where(self.log_triggers == start_trigger)[0]
        for idx in starts_log[:-1]:
            issues.append(
                self._issue(
                    issue='false_start_log',
                    severity='warning',
                    log_index=int(idx),
                    trigger=int(start_trigger),
                    message=(
                        f'Log contains start trigger {start_trigger} before '
                        'the final start trigger.'
                    ),
                    repair='fix_false_starts',
                )
            )

        starts_eeg = np.where(self.eeg_triggers == start_trigger)[0]
        for idx in starts_eeg[:-1]:
            sample = self.eeg_trigger_samples[idx][0]
            issues.append(
                self._issue(
                    issue='false_start_eeg',
                    severity='warning',
                    eeg_index=int(idx),
                    trigger=int(start_trigger),
                    sample=int(sample),
                    message=(
                        f'EEG contains start trigger {start_trigger} before '
                        'the final start trigger.'
                    ),
                    repair='fix_false_starts',
                )
            )

        return issues

    def _check_trigger_pulse_lengths(self, minlen=None, maxlen=None):
        """Report trigger pulses outside requested length bounds."""
        if minlen is None and maxlen is None:
            return []

        issues = []
        for idx, samples in enumerate(self._trigger_runs()):
            pulse_len = len(samples)
            trigger = self.trigger_channel[samples[0]]
            sample = samples[0]

            if minlen is not None and pulse_len < minlen:
                issues.append(
                    self._issue(
                        issue='short_trigger_pulse',
                        severity='warning',
                        eeg_index=int(idx),
                        trigger=int(trigger),
                        sample=int(sample),
                        message=(
                            f'EEG trigger pulse is {pulse_len} samples; '
                            f'expected at least {minlen}.'
                        ),
                        repair='repair_aberrant_trigger',
                    )
                )

            if maxlen is not None and pulse_len > maxlen:
                issues.append(
                    self._issue(
                        issue='long_trigger_pulse',
                        severity='warning',
                        eeg_index=int(idx),
                        trigger=int(trigger),
                        sample=int(sample),
                        message=(
                            f'EEG trigger pulse is {pulse_len} samples; '
                            f'expected at most {maxlen}.'
                        ),
                        repair='repair_aberrant_trigger',
                    )
                )

        return issues

    def _trigger_runs(self):
        """Return sample runs where the trigger channel is not at baseline."""
        mask = self.trigger_channel != self.trigger_default
        boundaries = np.flatnonzero(np.diff(self.trigger_channel) != 0) + 1
        runs = np.split(np.arange(self.trigger_channel.size), boundaries)
        return [run for run in runs if mask[run[0]]]

    def repair_aberrant_trigger(self, minlen=None, maxlen=None):
        """Replace suspicious trigger pulses with neighboring or default values.

        Pulses shorter than ``minlen`` or longer than ``maxlen`` are replaced.
        When an adjacent sample belongs to a neighboring trigger event, that
        neighboring trigger code is used; otherwise the session's trigger
        default is used.

        Args:
            minlen (int, optional): Repair pulses shorter than this many samples.
            maxlen (int, optional): Repair pulses longer than this many samples.
        """
        self._refresh_diagnostic_events()
        runs = self._trigger_runs()

        for idx, samples in enumerate(runs):
            pulse_len = len(samples)
            too_short = minlen is not None and pulse_len < minlen
            too_long = maxlen is not None and pulse_len > maxlen
            if not (too_short or too_long):
                continue

            replacement = self.trigger_default
            if idx > 0 and samples[0] > 0:
                previous_sample = samples[0] - 1
                if self.trigger_channel[previous_sample] != self.trigger_default:
                    replacement = self.trigger_channel[previous_sample]
            if replacement == self.trigger_default and samples[-1] + 1 < self.trigger_channel.size:
                next_sample = samples[-1] + 1
                if self.trigger_channel[next_sample] != self.trigger_default:
                    replacement = self.trigger_channel[next_sample]

            self.trigger_channel[samples] = replacement

        self.get_eeg_events(verbose=False)

    def fix_false_starts(self, start_trigger=1):
        """Discard log and EEG events before the last start trigger.

        Args:
            start_trigger (int): Trigger code marking the real experiment start.
        """
        self._refresh_diagnostic_events()

        starts_log = self.find_event({self.trigger_column: [start_trigger]})
        if len(starts_log) > 1:
            self.log = self.log.iloc[starts_log[-1]:].copy()
            self.update_log()

        starts_eeg = np.where(self.eeg_triggers == start_trigger)[0]
        if len(starts_eeg) > 1:
            last_prestart_event = starts_eeg[-1] - 1
            if last_prestart_event >= 0:
                sample = self.eeg_trigger_samples[last_prestart_event][-1]
                self.trigger_channel[:sample + 1] = self.trigger_default

        self.get_eeg_events(verbose=False)

    def merge_recordings(self, other):
        """Merge another EEGSession into this one.

        This repair needs a separate design pass because it depends on how we
        want to reconcile headers, loaded channels, and log ownership.
        """
        raise NotImplementedError('merge_recordings is not implemented yet.')

    def insert_missing_eeg_triggers(self):
        """Insert EEG trigger codes present in the log but absent from EEG data.

        This is intentionally left for a later pass because it mutates timing
        and should be driven by diagnostics we trust first.
        """
        raise NotImplementedError('insert_missing_eeg_triggers is not implemented yet.')

    def insert_missing_bdf(self):
        """Backward-compatible alias for ``insert_missing_eeg_triggers``."""
        return self.insert_missing_eeg_triggers()
