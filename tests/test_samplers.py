"""Module to test the sampler."""

import pytest
import numpy as np

from audio_loader.features.raw_audio import WindowedAudio
from audio_loader.features.mfcc import WindowedMFCC
from audio_loader.ground_truth.timit import TimitGroundTruth
from audio_loader.ground_truth.timit import PHON
from audio_loader.samplers.windowed import WindowedSampler
from audio_loader.samplers.file import FileSampler


class TestWindowedSampler:
    """Test WindowedSampler."""

    def test_init(self, timit_like_path, timit_like_datapath, timit_like_gtpath):
        """Test instantiation."""
        gt_getter = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath)
        audio_extractor = WindowedAudio(1024, 512, 16000, normalize=True, padding=True)
        mfcc_extractor = WindowedMFCC(1024, 512, 16000, n_mfcc=32, normalize=True)
        try:
            WindowedSampler([audio_extractor], gt_getter, 10000)
            WindowedSampler([audio_extractor, mfcc_extractor], gt_getter, 10000)
        except Exception as exception:
            pytest.fail(f"Unexpected  error: {exception}")

    def test_get_samples_from(self, timit_like_path, timit_like_datapath, timit_like_gtpath):
        """Test sampling iterator."""
        gt_getter = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath)
        audio_extractor = WindowedAudio(1024, 512, 16000, normalize=True, padding=True)
        mfcc_normalized = WindowedMFCC(1024, 512, 16000, n_mfcc=32, normalize=True)

        sampler = WindowedSampler([audio_extractor], gt_getter, 10000)
        iterator = sampler.get_samples_from("train")
        sample, ground_truth = next(iterator)
        assert np.sum(ground_truth) == 1
        assert sample.shape == (1, 18, 1024)
        assert np.isclose(ground_truth[gt_getter.get_index_from("h#")], .97, atol=0.01)
        assert np.isclose(ground_truth[gt_getter.get_index_from("p")], .03, atol=0.01)

        sampler = WindowedSampler([audio_extractor, mfcc_normalized], gt_getter, 10000)
        iterator = sampler.get_samples_from("train")
        sample, ground_truth = next(iterator)
        assert sample.shape == (1, 18, 1056)

        # consume all iterator
        for _ in iterator:
            pass

        audio_extractor_no_pad = WindowedAudio(1024, 512, 16000, normalize=True, padding=False)
        sampler = WindowedSampler([audio_extractor_no_pad], gt_getter, 10000)
        iterator = sampler.get_samples_from("train")
        sample, ground_truth = next(iterator)
        assert np.sum(ground_truth) == 1
        assert sample.shape == (1, 18, 1024)
        print(ground_truth)
        assert np.isclose(ground_truth[gt_getter.get_index_from("h#")], 0.93, atol=0.01)
        assert np.isclose(ground_truth[gt_getter.get_index_from("p")], 0.05, atol=0.01)
        assert np.isclose(ground_truth[gt_getter.get_index_from("iy")], 0.02, atol=0.01)

    def test_filepath(self, timit_like_path, timit_like_datapath, timit_like_gtpath):
        """Test getting filepath along of the batch."""
        gt_getter = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath)
        audio_extractor = WindowedAudio(1024, 512, 16000, normalize=True, padding=True)
        sampler = WindowedSampler([audio_extractor], gt_getter, 4000, output_filepath=True)
        iterator = sampler.get_samples_from("train")
        (_, _), filepath = next(iterator)
        assert filepath == "train/R1/S1_1.WAV"

    def test_get_output_description(self, timit_like_path, timit_like_datapath, timit_like_gtpath):
        """Test the description methods."""
        gt_getter = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath)
        audio_extractor = WindowedAudio(1024, 512, 16000, normalize=True, padding=True)
        mfcc_normalized = WindowedMFCC(1024, 512, 16000, n_mfcc=32, normalize=True)

        sampler = WindowedSampler([audio_extractor, mfcc_normalized], gt_getter, 1536)
        expected_result = {
            "samples": {
                "WindowedAudio": list(range(0, 1024)),
                "WindowedMFCC": list(range(1024, 1056))
            },
            "ground_truth": PHON
        }

        assert sampler.get_output_description() == expected_result


class TestFileSampler:
    """Test FileSampler class."""

    def test_init(self, timit_like_path, timit_like_datapath, timit_like_gtpath):
        """Test instantiation."""
        gt_getter = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath)
        audio_extractor = WindowedAudio(1024, 512, 16000, normalize=True, padding=True)
        mfcc_extractor = WindowedMFCC(1024, 512, 16000, n_mfcc=32, normalize=True)
        try:
            FileSampler([audio_extractor], gt_getter)
            FileSampler([audio_extractor, mfcc_extractor], gt_getter)
        except Exception as exception:
            pytest.fail(f"Unexpected  error: {exception}")

    def test_get_samples_from(self, timit_like_path, timit_like_datapath, timit_like_gtpath):
        """Test sampling iterator."""
        gt_getter = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath)
        audio_extractor = WindowedAudio(1024, 512, 16000, normalize=True, padding=True)
        mfcc_normalized = WindowedMFCC(1024, 512, 16000, n_mfcc=32, normalize=True)

        sampler = FileSampler([audio_extractor], gt_getter)
        iterator = sampler.get_samples_from("train")
        sample, ground_truth = next(iterator)
        assert np.isclose(np.sum(ground_truth), 0.99, atol=0.01)
        assert sample.shape == (1, 130, 1024)

        sampler = FileSampler([audio_extractor, mfcc_normalized], gt_getter)
        iterator = sampler.get_samples_from("train")
        sample, ground_truth = next(iterator)
        assert sample.shape == (1, 130, 1056)

        # consume all iterator
        for _ in iterator:
            pass

        audio_extractor_no_pad = WindowedAudio(1024, 512, 16000, normalize=True, padding=False)
        sampler = FileSampler([audio_extractor_no_pad], gt_getter, 4000)
        iterator = sampler.get_samples_from("train")
        sample, ground_truth = next(iterator)
        assert np.isclose(np.sum(ground_truth), 0.99, atol=0.01)
        assert sample.shape == (1, 128, 1024)

    def test_get_output_description(self, timit_like_path, timit_like_datapath, timit_like_gtpath):
        """Test the description methods."""
        gt_getter = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath)
        audio_extractor = WindowedAudio(1024, 512, 16000, normalize=True, padding=True)
        mfcc_normalized = WindowedMFCC(1024, 512, 16000, n_mfcc=32, normalize=True)

        sampler = FileSampler([audio_extractor, mfcc_normalized], gt_getter)
        expected_result = {
            "samples": {
                "WindowedAudio": list(range(0, 1024)),
                "WindowedMFCC": list(range(1024, 1056))
            },
            "ground_truth": PHON
        }

        assert sampler.get_output_description() == expected_result
