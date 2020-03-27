"""Module to define all abstract class for feature selection."""
import abc


class FeatureSelection(abc.ABC):
    """Abstract class for all activity detection."""

    def __init__(self, win_size, hop_size, sampling_rate, padding=True):
        """Default init.

        Parameters
        ----------
        win_size: int
            Number of samples to use for the window size.

        hop_size: int
            Number of samples to use for hopping windows.

        sampling_rate: int
            Sampling rate expected by the signal in the process method.

        padding: boolean, optional
            True when process will apply padding and False if not.
        """
        self.win_size = win_size
        self.hop_size = hop_size
        self.sampling_rate = sampling_rate
        self.padding = padding

    @abc.abstractmethod
    def process(self, signal):
        """Return the filter for features."""
