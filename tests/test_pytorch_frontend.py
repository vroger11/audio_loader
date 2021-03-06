"""Module to test dl_frontend submodules."""

import pytest

from audio_loader.features.raw_audio import WindowedAudio
from audio_loader.features.mfcc import WindowedMFCC
from audio_loader.ground_truth.timit import TimitGroundTruth
from audio_loader.dl_frontends.pytorch.fill_ram import get_dataloader_fixed_size
from audio_loader.samplers.windowed_segments import WindowedSegmentSampler


class TestsPytorch:
    """Class to test every frontends functions designed for PyTorch."""
    def test_fill_cpu_ram(self, timit_like_path, timit_like_datapath, timit_like_gtpath):
        """Test the fill cpu rarm functionality."""
        gt_getter = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath)
        audio_extractor = WindowedAudio(1024, 512, 16000, normalize=True, padding=True)
        mfcc_extractor = WindowedMFCC(1024, 512, 16000, n_mfcc=32, normalize=True)
        sampler = WindowedSegmentSampler([audio_extractor, mfcc_extractor], gt_getter, 8000)
        try:
            get_dataloader_fixed_size(sampler, 32, "train")
            get_dataloader_fixed_size(sampler, 16, "test")
        except Exception as exception:
            pytest.fail(f"Unexpected  error: {exception}")
