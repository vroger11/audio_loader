import librosa
import numpy as np

from audio_loader.features.feature_extractor import FeatureExtractor


class WindowedMFCC(FeatureExtractor):
    """Get windowed MFC coefficients."""

    def __init__(self, win_size, hop_size, sampling_rate, n_mfcc, normalize=True, delta_orders=[]):
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

        delta_orders: list, optional
            list of delta orders to add to the output
        """
        # librosa padding is always True
        super().__init__(win_size, hop_size, sampling_rate, padding=True, delta_orders=delta_orders)
        self.n_mfcc = n_mfcc

    def process(self, signal, sampling_rate):
        """Compute windowed MFCC from the signal.

        Parameters
        ----------
        signal: array
            Shape should be (raw, channel).

        sampling_rate: int
            Sampling rate of signal.
        """
        self.check_sampling_rate(sampling_rate)

        def compute_mfcc(signal):
            """Compute MFCC with fixed parameters and add deltas at the end."""
            features = librosa.feature.mfcc(
                signal, sampling_rate, n_mfcc=self.n_mfcc, dct_type=2,
                norm='ortho', n_fft=self.win_size, hop_length=self.hop_size, htk=False,
                center=True, pad_mode='reflect'
            )

            deltas = []
            # compute deltas
            for order in self.delta_orders:
                deltas.append(
                    librosa.feature.delta(features, order=order, width=5, mode='mirror')
                )

            # concatenate deltas
            for delta in deltas:
                features = np.concatenate((features, delta), axis=0)

            return features

        # computation of the first channel
        signal = signal.T
        features = compute_mfcc(signal[0])

        # computation and reshape of the next channels
        features = features.reshape(1, *features.shape)
        for i in range(1, signal.shape[0]):
            features = np.concatenate(
                (features,
                 compute_mfcc(signal[i]).reshape(features.shape))
            )


        if self.normalize:
            features = features/np.absolute(features).max()

        return features.transpose(0, 2, 1)

    def size_last_dim(self):
        """Return the size of the last dimension when process a signal."""
        return self.n_mfcc * (len(self.delta_orders) + 1)
