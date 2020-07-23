import librosa
import numpy as np

from audio_loader.features.feature_extractor import FeatureExtractor


class WindowedLogMel(FeatureExtractor):
    """Get windowed Mel Spectrogram."""

    def __init__(self, win_size, hop_size, sampling_rate, n_mels, normalize=True):
        """Initialize the parameters to compute.

        Parameters
        ----------
        win_size: integer
            Number of samples to take.

        hop_size: int
            Number of samples to jump from the current cursor.

        sampling_rate: integer
            Expected sampling rate by the extractor.

        n_mels: integer
            Number of Mel bands to generate.

        normalize: boolean, optional
            Normalize the data between 0 and 1.
        """
        # librosa padding is always True
        super().__init__(win_size, hop_size, sampling_rate, padding=True)
        self.n_mels = n_mels
        self.filter = librosa.filters.mel(sampling_rate, win_size, self.n_mels)

    def process(self, signal, sampling_rate):
        """Compute windowed Log Mel spectrogram from the signal.

        Parameters
        ----------
        signal: array
            Shape should be (raw, channel).

        sampling_rate: int
            Sampling rate of signal.
        """
        self.check_sampling_rate(sampling_rate)

        def compute_mel(signal):
            """Compute Log Mel Spectrogram with fixed parameters."""
            mel_spectrogram = librosa.feature.melspectrogram(
                signal, sr=self.sampling_rate,
                n_fft=self.win_size, hop_length=self.hop_size,
                n_mels=self.n_mels)
            return librosa.power_to_db(mel_spectrogram, ref=np.max)

        # computation of the first channel
        signal = signal.T
        features = compute_mel(signal[0])

        # computation and reshape of the next channels
        features = features.reshape(1, *features.shape)
        for i in range(1, signal.shape[0]):
            features = np.concatenate(
                (features,
                 compute_mel(signal[i]).reshape(features.shape))
            )


        if self.normalize:
            features = features/np.absolute(features).max()

        return features.transpose(0, 2, 1)

    def size_last_dim(self):
        """Return the size of the last dimension when process a signal."""
        return self.n_mels
