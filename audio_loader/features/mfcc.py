import librosa
import numpy as np

from audio_loader.features.feature_extractor import FeatureExtractor


class WindowedMFCC(FeatureExtractor):
    """Get windowded MFC coeficients."""

    def __init__(self, win_size, hop_size, sampling_rate, n_mfcc, normalize=True):
        """Initialize the parameters to compute.

        Parameters
        ----------
        win_size: integer
            Number of samples to take.

        hop_size: int
            Number of samples to jump from the current cursor.

        sampling_rate: integer
            Expected sampling rate by the extractor.

        n_mfcc: integer
            Number of MFCCs to return.

        normalize: boolean, optional
            Normalize the data between 0 and 1.
        """
        # librosa padding is always True
        super().__init__(win_size, hop_size, sampling_rate, padding=True)
        self.n_mfcc = n_mfcc

    def process(self, signal, sampling_rate):
        """Compute windowed mfcc from the signal.

        Parameters:
        -----------

        signal: array
            Shape should be (raw, channel).

        sampling_rate: int
            Sampling rate of signal.
        """
        self.check_sampling_rate(sampling_rate)

        signal = signal.T
        features = librosa.feature.mfcc(
            signal[0], sampling_rate, n_mfcc=self.n_mfcc, dct_type=2,
            norm='ortho', n_fft=self.win_size, hop_length=self.hop_size, htk=False,
            center=True, pad_mode='reflect'
        )

        features = features.reshape(1, *features.shape)
        for i in range(1, signal.shape[0]):
            features = np.concatenate(
                (features,
                 librosa.feature.mfcc(
                     signal[i], sampling_rate, n_mfcc=self.n_mfcc, dct_type=2,
                     norm='ortho', n_fft=self.win_size, hop_length=self.hop_size, htk=False,
                     center=True, pad_mode='reflect'
                 ).reshape(features.shape))
            )


        if self.normalize:
            features = features/np.absolute(features).max()

        return features.transpose(0, 2, 1)

    def size_last_dim(self):
        """Return the size of the last dimension when process a signal."""
        return self.n_mfcc
