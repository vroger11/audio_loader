"""Module that implement a dynamic sampler linked to groundtruth start and end times."""
from os.path import join
from random import shuffle

import numpy as np
import soundfile as sf

from audio_loader.samplers.decorator import SamplerBase


class DynamicSampler(SamplerBase):
    """Create samples with associated groundtruth.

    Data outputed are segments with different sizes.
    The size depends to groundtruth start and end time for each ground truth samples.
    """

    def __init__(self, feature_processors, groundtruth, supervised=True,
                 output_filepath=False, activity_detection=None):
        """Initialize the sampler.

        Parameters
        ----------
        feature_processors: list
            List of feature processors.

        groundtruth:
            Contains methods allowing to get all sets + groundtruth.

        supervised: boolean
            Return the groundthruth alongside with each sample.

        output_filepath: boolean, optional
            add the filepath used to the output of the sampler
        """
        super().__init__(feature_processors, groundtruth,
                         supervised=supervised, output_filepath=output_filepath)


    def get_samples_from(self, selected_set, randomize_files=False):
        """Iterator other the ground truth."""
        file_list = self.get_file_list(selected_set)
        if randomize_files:
            file_list = file_list.copy()  # the shuffle should not impact the initial list
            shuffle(file_list)

        for filepath in file_list:
            signal, sr = sf.read(join(self.groundtruth.data_folderpath, filepath),
                                 always_2d=True, dtype='float32')

            samples_times = self.groundtruth.get_samples_time_in(filepath)
            for sample_begin, sample_end in samples_times:
                # get sample
                x = None
                sample_raw_data = signal[sample_begin:sample_end]
                for feature_processor in self.feature_processors:
                    if x is None:
                        x = feature_processor.process(sample_raw_data, sr)
                    else:
                        x = np.concatenate((x, feature_processor.process(signal, sr)), axis=2)

                if self.activity_detection is not None:
                   raise Exception("Dynamic sampler and VAD are not yet implemented")

                # prepare output
                if self.supervised:
                    y = self.groundtruth.get_gt_at_sample(filepath, sample_begin, sample_end)
                    sample = (x, y)
                else:
                    sample = tuple(x)

                if self.output_filepath:
                    yield (sample, filepath)
                else:
                    yield sample
