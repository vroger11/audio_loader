"""Module to test audio_loader.features submodules."""

import pytest

from audio_loader.features.raw_audio import WindowedAudio
from audio_loader.features.mfcc import WindowedMFCC


class TestRawAudio:
    """Test the raw_audio module."""

    def test_windowed(self, testing_audio):
        """Test the WindowedAudio class."""
        signal, samplerate = testing_audio
        try:
            feature_extractor = WindowedAudio(1024, 512, 16000, normalize=True, padding=True)
            raw_audio_padded = feature_extractor.process(signal, samplerate)

            feature_extractor = WindowedAudio(1024, 512, 16000, normalize=True, padding=False)
            raw_audio_not_padded = feature_extractor.process(signal, samplerate)

            feature_extractor = WindowedAudio(1024, 1024, 16000, normalize=True, padding=True)
            raw_audio_padded_no = feature_extractor.process(signal, samplerate)

            feature_extractor = WindowedAudio(1024, 1024, 16000, normalize=True, padding=False)
            raw_audio_not_padded_no = feature_extractor.process(signal, samplerate)
        except Exception as exception:
            pytest.fail(f"Unexpected  error: {exception}")

        assert raw_audio_padded.shape == (1, 1 + int((len(signal) - 1024 + 1024) / 512), 1024)
        assert raw_audio_not_padded.shape == (1, 1 + int((len(signal) - 1024) / 512), 1024)
        assert raw_audio_padded_no.shape == (1, 1 + int((len(signal) - 1024 + 1024) / 1024), 1024)
        assert raw_audio_not_padded_no.shape == (1, 1 + int((len(signal) - 1024) / 1024), 1024)
        assert (raw_audio_padded.max() <= 1) and (raw_audio_padded.min() >= -1)
        assert (raw_audio_not_padded.max() <= 1) and (raw_audio_not_padded.min() >= -1)
        assert (raw_audio_padded_no.max() <= 1) and (raw_audio_padded_no.min() >= -1)


class TestMFCC:
    """Test the mfcc module."""

    def test_windowed(self, testing_audio):
        """Test the WindowedMFCC class."""
        signal, samplerate = testing_audio
        try:
            feature_extractor = WindowedMFCC(1024, 512, 16000, n_mfcc=32, normalize=True)
            mfcc_normalized = feature_extractor.process(signal, samplerate)
            feature_extractor_no = WindowedMFCC(1024, 1024, 16000, n_mfcc=32, normalize=True)
            mfcc_no_overlap = feature_extractor_no.process(signal, samplerate)

        except Exception as exception:
            pytest.fail(f"Unexpected  error: {exception}")

        assert (mfcc_normalized.max() <= 1) and (mfcc_normalized.min() >= -1)
        assert mfcc_normalized.shape[-1] == 32
        assert mfcc_normalized.shape[-2] == 1 + int((len(signal) - 1024 + 1024) / 512)
        assert mfcc_no_overlap.shape[-1] == 32
        assert mfcc_no_overlap.shape[-2] == 1 + int((len(signal) - 1024 + 1024) / 1024)
