"""Module that implement an iterator version of the sampler."""
from os.path import join
from random import shuffle

import numpy as np
import soundfile as sf

from audio_loader.samplers.decorator import SamplerBase


class WindowedSampler(SamplerBase):
    """Create samples with associated groundtruth.

    No cache involved.
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

            x = None
            for feature_processor in self.feature_processors:
                if x is None:
                    x = feature_processor.process(signal, sr)
                else:
                    x = np.concatenate((x, feature_processor.process(signal, sr)), axis=2)

            if self.supervised:
                y = self.groundtruth.get_gt_from_sample(filepath,
                                                        self.fe_win_size,
                                                        self.fe_hop_size,
                                                        padding=self.fe_padding)

            if self.activity_detection is not None:
                fe_filter = self.activity_detection.process(signal)
                x = x[fe_filter]
                if len(x.shape) != 3:
                    x = x.reshape(1, *x.shape)

                # start FIXME y does not have a channel dimension
                fe_filter = fe_filter.all(axis=0)
                # end FIXME
                y = y[fe_filter]

            nb_win = x.shape[1]
            n_frames = 1 + int((nb_win - self.n_frames_select) / self.n_frames_hop)
            for i in range(0, n_frames):
                i = i*self.n_frames_hop
                x_selected = x[:, i:i+self.n_frames_select, :]
                if self.supervised:
                    y_selected = y[i:i+self.n_frames_select, :]
                    sample = (x_selected, np.sum(y_selected, axis=0)/self.n_frames_select)
                else:
                    sample = tuple(x_selected)

                if self.output_filepath:
                    yield (sample, filepath)
                else:
                    yield sample
