"""Decorator that every sampler should inherit."""
import abc


class SamplerBase:
    """Decorator for all samplers."""

    def __init__(self, feature_processors, groundtruth, seg_size, overlap=0.5, supervised=True,
                 output_filepath=False, activity_detection=None):
        """Initialise the sampler.

        Parameters
        ----------
        feature_processors: list
            List of feature processors.

        groundtruth:
            Contains methods allowing to get all sets + groundtruth.

        seg_size: integer (greather than 0)
            Size of segments in number of samples.

        overlap: float between 0. and 1.
            Overlap of the segments.

        supervised: boolean
            Return the groundthruth alongside with each sample.
        """
        self.feature_processors = feature_processors
        self.groundtruth = groundtruth
        self.supervised = supervised
        self.output_filepath = output_filepath

        self.fe_win_size = feature_processors[0].win_size
        if self.fe_win_size > seg_size:
            raise Exception("seg_size should be lager or equel to feature extractors win_size")

        self.fe_hop_size = feature_processors[0].hop_size
        self.fe_padding = feature_processors[0].padding
        for feature_processor in self.feature_processors:
            if feature_processor.win_size != self.fe_win_size or feature_processor.hop_size != self.fe_hop_size:
                raise Exception(
                    "All feature processors should have the same win_size and hop_size.")
            if feature_processor.padding != self.fe_padding:
                raise Exception("All feature processors should have the same padding value.")

        self.n_frames_select = 1 + int((seg_size - self.fe_win_size) / self.fe_hop_size)
        self.n_frames_hop = int(self.n_frames_select * (1 - overlap))
        if self.n_frames_hop < 1:
            raise Exception(
                f"seg_size {seg_size} is too small for the chosen extractor(s)")

        self.seg_size = self.fe_win_size + (self.n_frames_select-1) * self.fe_hop_size
        self.hop_seg_size = self.fe_win_size + (self.n_frames_hop-1) * self.fe_hop_size

        self.activity_detection = activity_detection
        if self.activity_detection is not None:
            if self.activity_detection.win_size != self.fe_win_size or self.activity_detection.hop_size != self.fe_hop_size:
                raise Exception(
                    f"Activity detection Win size and hop size should be the same as for feature extractors.")

            if self.activity_detection.padding != self.fe_padding:
                raise Exception(
                    f"Activity detection padding should be the same as for feature extractors.")

            if self.activity_detection.sampling_rate != feature_processors[0].sampling_rate:
                raise Exception(
                    f"Activity detection padding should have the same expected sampling rate as the feature processors.")

    @abc.abstractmethod
    def get_samples_from(self, selected_set, randomize_files=False):
        """Get features and ground truth (if suppervised) from selected_set list."""

    def get_file_list(self, selected_set):
        """Get a list of filepaths corresponding to a given set.

        Parameters
        ----------
        selected_set: str
            Set to use, can be ever "train", "dev" or "test"

        Return
        ------
        A list of filepath corresponding to selected_set.
        """
        if selected_set in ('train', 'training'):
            return self.groundtruth.training_set

        if selected_set in ('valid', 'validation', 'dev', "devel"):
            return self.groundtruth.validation_set

        if selected_set in ('test', "testing"):
            return self.groundtruth.testing_set

        raise Exception(f"{selected_set} set not recognized.")

    def get_output_description(self):
        """Return a description of each sample using a dictionnary."""
        desc = {}
        desc["samples"] = {}
        i = 0
        for feature_processor in self.feature_processors:
            proc_class_name = feature_processor.__class__.__name__
            last_dim_size = feature_processor.size_last_dim()
            value_range = list(range(i, i + last_dim_size))
            desc["samples"][proc_class_name] = value_range
            i += last_dim_size

        if self.supervised:
            desc["ground_truth"] = self.groundtruth.get_output_description()
        else:
            desc["ground_truth"] = None

        return desc
