"""Module providing all constants and fixtures for testing."""

from os.path import join, dirname
import pytest
import soundfile as sf

TESTING_DATASET_PATH = join(dirname(__file__), "testing_datasets")
TEST_WAVEFILE_PATH = join(TESTING_DATASET_PATH, "TIMIT_like/data/train/R1/S1_1.WAV")

@pytest.fixture(scope="module")
def testing_audio_path():
    """Path of the audio that helps for testing."""
    return TEST_WAVEFILE_PATH

@pytest.fixture(scope="module")
def testing_audio():
    """Load data for the tests."""
    return sf.read(TEST_WAVEFILE_PATH, always_2d=True, dtype='float32')

# TIMIT like variables

@pytest.fixture(scope="module")
def timit_like_path():
    """Path to the timit_like dataset."""
    return join(TESTING_DATASET_PATH, "TIMIT_like")

@pytest.fixture(scope="module")
def timit_like_datapath():
    """Data folder path."""
    return join(TESTING_DATASET_PATH, "TIMIT_like/data")

@pytest.fixture(scope="module")
def timit_like_gtpath():
    """Ground truth folder path."""
    return join(TESTING_DATASET_PATH, "TIMIT_like/ground_truth")

# C2SI like variables

@pytest.fixture(scope="module")
def c2si_like_path():
    """Path to the timit_like dataset."""
    return join(TESTING_DATASET_PATH, "C2SI_like/")

@pytest.fixture(scope="module")
def c2si_like_gtpath():
    """Ground truth file path for ."""
    return join(TESTING_DATASET_PATH, "C2SI_like/ground_truth.xlsx")
