"""Backwards-compatible wrappers for session repair functions.

These operations are now available as methods on :class:`SessionHandler`.
This module is kept for backwards compatibility with existing scripts.
"""

from .recording_manager import SessionHandler


def repair_aberrant_trigger(raw, header, channel_headers, log, maxlen):
    """See :meth:`SessionHandler.repair_aberrant_trigger`."""
    session = SessionHandler(log, raw, channel_headers, header)
    session.repair_aberrant_trigger(maxlen)
    return session.return_data()


def fix_false_starts(raw, header, channel_headers, log):
    """See :meth:`SessionHandler.fix_false_starts`."""
    session = SessionHandler(log, raw, channel_headers, header)
    session.fix_false_starts()
    return session.return_data()


def merge_recordings(raw_1, raw_2, header_1, header_2,
                     channel_headers_1, channel_headers_2, log):
    """See :meth:`SessionHandler.merge_recordings`."""
    session = SessionHandler.merge_recordings(
        raw_1, raw_2, header_1, header_2,
        channel_headers_1, channel_headers_2, log,
    )
    return session.return_data()


def insert_missing_bdf(raw, header, channel_headers, log):
    """See :meth:`SessionHandler.insert_missing_bdf`."""
    session = SessionHandler(log, raw, channel_headers, header)
    session.insert_missing_bdf()
    return session.return_data()
