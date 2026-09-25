"""Backward-compatible API; new code should import from :mod:`covatad`."""

from covatad import CoVATabularDetector, TabICLEpistemicDetector

__all__ = ["CoVATabularDetector", "TabICLEpistemicDetector"]
