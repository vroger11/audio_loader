import librosa
import numpy as np

from audio_loader.features.feature_extractor import FeatureExtractor


class WindowedLogSpectrogram(FeatureExtractor):
    """Get windowed log spectrogram."""

    def __init__(self, win_size, hop_size, sampling_rate, normalize=True):
        """Initialize the parameters to compute.

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
        """
        # librosa padding is always True
        super().__init__(win_size, hop_size, sampling_rate, padding=True, normalize=normalize)

    def process(self, signal, sampling_rate):
        """Compute windowed spectrogram from the signal.

        Parameters
        ----------
        signal: array
            Shape should be (raw, channel).

        sampling_rate: int
            Sampling rate of signal.
        """
        self.check_sampling_rate(sampling_rate)

        def compute_log_spectrogram(signal):
            """Compute log spectrogram for a 1 channel Use of stft with fixed parameters."""
            spectrogram = np.abs(
                librosa.stft(signal, n_fft=self.win_size,
                             window='hann', hop_length=self.hop_size,
                             center=True, pad_mode='reflect'))

            return librosa.amplitude_to_db(spectrogram, ref=np.max)

        # computation of the first channel
        signal = signal.T
        features = compute_log_spectrogram(signal[0])

        # computation and reshape of the next channels
        features = features.reshape(1, *features.shape)
        for i in range(1, signal.shape[0]):
            features = np.concatenate(
                (features,
                 compute_log_spectrogram(signal[i]).reshape(features.shape))
            )

        if self.normalize:
            features = features/np.absolute(features).max()

        return features.transpose(0, 2, 1)

    def size_last_dim(self):
        """Return the size of the last dimension when process a signal."""
        return 1 + self.win_size/2
