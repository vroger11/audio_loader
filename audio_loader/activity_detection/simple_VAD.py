"""Module to implement a simple feature selection system based on thresholds over
energy and spectral flatness."""
import librosa
import numpy as np

from audio_loader.activity_detection.feature_selection import FeatureSelection


class Simple(FeatureSelection):
    """Simple voice activity detection, based on signal energy and spectral flatness."""

    def __init__(self,
                 win_size,
                 hop_size,
                 sampling_rate,
                 energy_threshold=0.2,
                 spectral_flatness_threshold=0.3,
                 smooth=5):
        """Initializes activity detector.

        Parameters
        ----------
        win_size: int
            Number of samples to use for the window size.

        hop_size: int
            Number of samples to use for hopping windows.

        sampling_rate: int
            Sampling rate expected by the signal in the process method.

        energy_threshold: float, optional
            Between 0. and 1..

        spectral_flatness_threshold: float, optional
            Between 0. and 1..

        smooth: int, optional
            Number of allowed filled holes.
        """
        super().__init__(win_size, hop_size, sampling_rate, padding=True)
        self.energy_threshold = energy_threshold
        self.spectral_flatness_threshold = spectral_flatness_threshold
        self.smooth = smooth

    def process(self, signal):
        """Executes the activity detection.

        Parameters
        ----------
            signal (array): 2d signal
                (n, channel)

        Return
        ------
            vector of activity
        """
        signal = signal.transpose(1, 0)

        res = []
        for channel in signal:
            # compute required features
            computed_energy = librosa.feature.rms(
                y=channel, frame_length=self.win_size, hop_length=self.hop_size)
            computed_spectrall_flatness = librosa.feature.spectral_flatness(
                y=channel, n_fft=self.win_size, hop_length=self.hop_size)

            # Voice Activity Detection
            energy95p = np.percentile(computed_energy, 95)
            if energy95p == 0:
                raise Exception(f"The channel is silent")

            normalized_en = (computed_energy / energy95p)
            out = np.logical_and(
                normalized_en > self.energy_threshold,
                computed_spectrall_flatness < self.spectral_flatness_threshold
            )

            if self.smooth > 0:
                start = -self.smooth
                for i in range(out.shape[1]):
                    if out[:, i]:
                        if (start != i-1) and (i-start < self.smooth):
                            out[:, start+1:i] = True

                        start = i

            res.append(out.flatten())

        return np.array(res)
