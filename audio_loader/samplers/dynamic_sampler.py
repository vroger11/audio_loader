"""Module that implement a dynamic sampler linked to groundtruth start and end times."""
from os.path import join
from random import shuffle

import numpy as np
import soundfile as sf

from audio_loader.samplers.decorator import SamplerBase


class DynamicSamplerFromGt(SamplerBase):
    """Create samples with associated groundtruth.

    The selection of the samples is defined by the ground truth.

    Data outputed are segments with different sizes.
    The size depends to groundtruth start and end time for each ground truth samples.
    """

    def get_samples_from(self, selected_set, randomize_files=False):
        """Iterator other the ground truth."""
        file_list = self.get_file_list(selected_set)
        if randomize_files:
            file_list = file_list.copy()  # the shuffle should not impact the initial list
            shuffle(file_list)

        for filepath in file_list:
            signal, sr = sf.read(join(self.groundtruth.data_folderpath, filepath),
                                 always_2d=True, dtype='float32')

            samples_times = self.groundtruth.get_gt_for(filepath)
            for sample_tuple in samples_times:
                if len(sample_tuple):
                    sample_begin, sample_end, y = sample_tuple
                else:
                    sample_begin, sample_end = sample_tuple

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
                    sample = (x, y)
                else:
                    sample = tuple(x)

                if self.output_filepath:
                    yield (sample, filepath)
                else:
                    yield sample


class DynamicSamplerFromAD(SamplerBase):
    """Create samples with associated groundtruth.

    The selection of the samples is defined by the activity detection.

    Data outputed are segments with different sizes.
    The size depends to groundtruth start and end time for each ground truth samples.
    """

    def get_samples_from(self, selected_set, randomize_files=False, merge_gt=False):
        """Iterator other the ground truth.

        merge_gt: bool
            Merge ground truth over yielded segments to have one vector instead of a matrix.
        """
        file_list = self.get_file_list(selected_set)
        if randomize_files:
            file_list = file_list.copy()  # the shuffle should not impact the initial list
            shuffle(file_list)

        for filepath in file_list:
            signal, sr = sf.read(join(self.groundtruth.data_folderpath, filepath),
                                 always_2d=True, dtype='float32')

            if self.supervised:
                y = self.groundtruth.get_gt_from_sample(filepath,
                                                        self.fe_win_size,
                                                        self.fe_hop_size,
                                                        padding=self.fe_padding)

            # get sample
            x = None
            for feature_processor in self.feature_processors:
                if x is None:
                    x = feature_processor.process(signal, sr)
                else:
                    x = np.concatenate((x, feature_processor.process(signal, sr)), axis=2)

            if self.activity_detection is not None:
                fe_filter = self.activity_detection.process(signal)

                # FIXME add support for multi channels
                fe_filter = fe_filter[0] # simple hack to make it work for single channel
                indices_filtered = np.where(fe_filter)[0]
                indices = []
                prev = indices_filtered[0]
                start = indices_filtered[0]
                for i in range(1, len(indices_filtered)):
                    if prev+1 == indices_filtered[i]:
                        prev = prev+1
                    else:
                        indices.append((start, prev+1))
                        start = indices_filtered[i]
                        prev = indices_filtered[i]

                for i, ik in indices:
                    x_selected = x[0][i:ik] # FIXME multi channel
                    if len(x_selected.shape) != 3:
                        x_selected = x_selected.reshape(1, *x_selected.shape)

                    # prepare output
                    if self.supervised:
                        # start FIXME y does not have a channel dimension
                        y_selected = y[i:ik]

                        if merge_gt:
                            y_selected = y_selected.mean(axis=0)

                        sample = (x_selected, y_selected)
                    else:
                        sample = tuple(x_selected)

                    if self.output_filepath:
                        yield (sample, filepath)
                    else:
                        yield sample

            else:
                # prepare output
                if self.supervised:
                    if merge_gt:
                        y = y.mean(axis=0)

                    sample = (x, y)
                else:
                    sample = tuple(x)

                if self.output_filepath:
                    yield (sample, filepath)
                else:
                    yield sample
