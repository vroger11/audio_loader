import librosa
import numpy as np

from audio_loader.features.feature_extractor import FeatureExtractor


class WindowedAudio(FeatureExtractor):
    """Get windowed and padded audios."""

    def process(self, signal, sampling_rate):
        """Compute windowed signal from the audio in filepath.

        Parameters
        ----------
        signal: array
            Shape should be (raw, channel).

        sampling_rate: int
            Sampling rate of signal.
        """
        self.check_sampling_rate(sampling_rate)

        if self.normalize:
            signal = signal / np.max(np.abs(signal))

        if self.padding:
            # apply librosa default padding
            signal = np.pad(signal, ((self.win_size//2, self.win_size//2), (0, 0)), mode='reflect')

        signal = librosa.util.frame(signal, self.win_size, self.hop_size, axis=0)
        return signal.transpose(2, 0, 1)

    def size_last_dim(self):
        """Return the size of the last dimension when process a signal."""
        return self.win_size
