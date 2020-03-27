"""Module that implement an iterator version of the sampler."""
from os.path import join
from random import shuffle

import numpy as np
import soundfile as sf

from audio_loader.samplers.decorator import SamplerBase


class FileSampler(SamplerBase):
    """Create samples with associated groundtruth.

    No cache involved.
    Samples does not have the same size. It depends on the size of the loaded signals.
    """

    def __init__(self, feature_processors, groundtruth, supervised=True, output_filepath=False):
        """Initialize the sampler."""
        # seg_size and overlap will not be used in this sampler
        super().__init__(feature_processors, groundtruth,
                         seg_size=feature_processors[0].win_size+1,
                         overlap=0, supervised=supervised,
                         output_filepath=output_filepath)

    def get_samples_from(self, selected_set, randomize_files=False):
        """Iterator other the ground truth."""
        file_list = self.get_file_list(selected_set)
        if randomize_files:
            file_list = file_list.copy()  # the shuffle should not impact the initial list
            shuffle(file_list)

        for filepath in file_list:
            signal, sr = sf.read(join(self.groundtruth.data_folderpath, filepath), always_2d=True, dtype='float32')

            x = None
            for feature_processor in self.feature_processors:
                if x is None:
                    x = feature_processor.process(signal, sr)
                else:
                    x = np.concatenate((x, feature_processor.process(signal, sr)), axis=2)

            if self.supervised:
                y = np.zeros(self.groundtruth.gt_size)
                id_audio = self.groundtruth.get_id(filepath)
                self.groundtruth._fill_output(id_audio, 0, len(signal), y)
                sample = (x, y)
            else:
                sample = tuple(x)

            if self.output_filepath:
                yield (sample, filepath)
            else:
                yield sample
