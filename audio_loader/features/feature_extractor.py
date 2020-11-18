"""Decorator of each feature extractor."""

import abc


class FeatureExtractor(abc.ABC):
    """Decorator for each feature extractor."""

    def __init__(self, win_size, hop_size, sampling_rate, normalize=True, padding=True, delta_orders=[], delta_width=5):
        """Create a feature extractor.

        Parameters
        ----------
        win_size: integer
            Number of samples to take.

        hop_size: int
            Number of samples to jump from the current cursor.

        sampling_rate: integer
            Expected sampling rate by the extractor.

        normalize: boolean, optional
            Normalize the data between 0 and 1.

        padding: boolean, optional
            center reflect padding (like librosa) if True.
            No padding if False.

        delta_orders: list, optional
            list of delta orders to add to the output

        delta_width: int
            Odd number (recomanded 5, 7 or 9)
        """
        self.win_size = win_size
        self.hop_size = hop_size
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.padding = padding
        self.delta_orders = delta_orders
        self.delta_width = delta_width

    @abc.abstractmethod
    def process(self, signal, sampling_rate):
        """Apply the feature extraction on signal."""

    @abc.abstractmethod
    def size_last_dim(self):
        """Return the size of the last dimension when process a signal."""

    def check_sampling_rate(self, sampling_rate):
        """Check if sampling_rate is the expected one."""
        if self.sampling_rate != sampling_rate:
            raise Exception(
                f"Bad samplerate: expected {self.sampling_rate} but got {sampling_rate}")
