"""Module to test the ground_truth submodules."""

import pytest
import numpy as np

from audio_loader.ground_truth.timit import TimitGroundTruth
from audio_loader.ground_truth.c2si import C2SI


class TestTimitFormat:
    """Test the TimitGroundTruth class."""

    def test_file_formats(self, timit_like_path, timit_like_datapath, timit_like_gtpath):
        """Test the two ways to acquire datapaths."""
        gt_getter_1 = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath)
        gt_getter_2 = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath,
                                       "ground_truth_simplified.csv")

        assert np.all(gt_getter_1.training_set == gt_getter_2.training_set)
        assert np.all(gt_getter_1.testing_set == gt_getter_2.testing_set)

    def test_get_gt_at_sample(self, timit_like_path, timit_like_datapath,
                              timit_like_gtpath, testing_audio_path):
        """Test get_gt_at_sample method."""
        gt_getter = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath)
        # tests for phoneme
        gt_getter.set_gt_format(phonetic=True, word=False, speaker_id=False)
        y = gt_getter.get_gt_at_sample(testing_audio_path, 3100, 8000)
        assert y[gt_getter.get_index_from("h#")] == 1.0
        assert np.sum(y) == 1.0
        y = gt_getter.get_gt_at_sample(testing_audio_path, 8000, 9000)
        assert y[gt_getter.get_index_from("h#")] == 0.83
        assert y[gt_getter.get_index_from("p")] == 0.17
        assert np.sum(y) == 1.0

    def test_get_gt_at_time(self, timit_like_path, timit_like_datapath,
                            timit_like_gtpath, testing_audio_path):
        """Test get_gt_at_time method."""
        gt_getter = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath)
        # tests for phoneme
        gt_getter.set_gt_format(phonetic=True, word=False, speaker_id=False)
        y = gt_getter.get_gt_at_time(testing_audio_path, 0.2, 0.25)
        assert y[gt_getter.get_index_from("h#")] == 1.0
        assert np.sum(y) == 1.0

    def test_get_gt_from_sample(self, timit_like_path, timit_like_datapath,
                                timit_like_gtpath, testing_audio_path, testing_audio):
        """Test getting ground truth windowing."""
        signal, _ = testing_audio
        gt_getter = TimitGroundTruth(timit_like_path, timit_like_datapath, timit_like_gtpath)
        # test for phoneme
        gt_getter.set_gt_format(phonetic=True, word=False, speaker_id=False)

        # with overlap
        y_pad = gt_getter.get_gt_from_sample(testing_audio_path, 512, 256, padding=True)
        y_no_pad = gt_getter.get_gt_from_sample(testing_audio_path, 512, 256, padding=False)
        windows_expected_pad = 1 + int((len(signal) - 512 + 512) / 256)
        windows_expected_no_pad = 1 + int((len(signal) - 512) / 256)
        assert y_pad.shape == (windows_expected_pad, gt_getter.gt_size)
        sum_gt = np.sum(y_pad, axis=1)
        assert np.array_equal(sum_gt[:], np.ones(windows_expected_pad))

        assert y_no_pad.shape == (windows_expected_no_pad, gt_getter.gt_size)
        sum_gt_no_pad = np.sum(y_no_pad, axis=1)
        assert np.array_equal(sum_gt_no_pad, np.ones(windows_expected_no_pad))

        # without overlap
        y_pad = gt_getter.get_gt_from_sample(testing_audio_path, 512, 512, padding=True)
        y_no_pad = gt_getter.get_gt_from_sample(testing_audio_path, 512, 512, padding=False)
        windows_expected_pad = 1 + int((len(signal) - 512 + 512) / 512)
        windows_expected_no_pad = 1 + int((len(signal) - 512) / 512)
        assert y_pad.shape == (windows_expected_pad, gt_getter.gt_size)
        sum_gt = np.sum(y_pad, axis=1)
        assert np.array_equal(sum_gt[:], np.ones(windows_expected_pad))

        assert y_no_pad.shape == (windows_expected_no_pad, gt_getter.gt_size)
        sum_gt_no_pad = np.sum(y_no_pad, axis=1)
        assert np.array_equal(sum_gt_no_pad, np.ones(windows_expected_no_pad))


class TestC2SI:
    """Test C2SI ground truth module."""

    def test_get_gt_at_sample(self, c2si_like_path, c2si_like_gtpath):
        """Test get_gt_at_sample method."""
        gt_getter = C2SI(c2si_like_path, "16k",
                         c2si_like_gtpath.replace(c2si_like_path, ""),
                         sets=None, severity_score=True)
        y = gt_getter.get_gt_at_sample("005000/TIS-005000-01-L01.wav", 0, 8000000000)
        assert np.isclose(y[0], 7.83, atol=0.01)
