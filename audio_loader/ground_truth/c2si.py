"""Tools helping with the C2SI corpus."""
import re

from glob import glob
from os.path import join

import numpy as np
import pandas as pd
import soundfile as sf

from audio_loader.ground_truth.challenge import Challenge


EXPLICIT_TASKS = ["maintained A", "reading", "describe an image", "spontaneus", "DAP", "Modality, Focus, Syntax or SVT"]
TASKS = ["V", "L", "D", "P", "N", "S"]
GROUPS = {
    "all": [1, 2],
    "patients": [1],
    "controls": [2]
}

class C2SI(Challenge):
    """Ground getter for C2SI dataset."""

    def __init__(self, c2si_root_folderpath, datapath="16k",
                 gtpath="BASE-Version-2020-01-05.xlsx", sets=None,
                 regression_scales=None,
                 severity_score=False,
                 intelligibility_score=False,
                 targeted_tasks=['L'],
                 group="all"):
        """Compatible with the C2SI corpus.

        Parameters:
        -----------
        c2si_root_folderpath: str
            Folder containing the C2SI data.

        datapath: str, optional
            Relative path of the data from the c2si_root_folderpath.

        gtpath: str, optional
            Relative path of the Exel file.

        sets: str, optional
            csv file which contains four columns:
                "relative path", "train", "dev", "test"
            if = None all data in data_folderpath are considered as a testing data

        regression_scales: list of intervals, optional
            Each scale give an output.

        severity_score: boolean, optional
            Output will be filled by the severity score.

        intelligibility_score: boolean, optional
            Output will be filled by the intelligibility score.

        targeted_tasks: list, optional
            List of tasks to use, possible values are "V", "L", "D", "P", "N", "S"
            "L" is for reading task.

        group: str, optional
            Corresponding group. Can be either "all", "patients" or "controls".
        """
        super().__init__(c2si_root_folderpath, datapath, gtpath)
        self.regression_scales = regression_scales
        self.severity_score = severity_score
        self.intelligibility_score = intelligibility_score
        if sets is not None:
            self.df_data = pd.read_csv(sets, delimiter=",")
        else:
            data_absolute_path = join(c2si_root_folderpath, self.datapath)
            dic_data = {"relative path": [], "train": [], "dev": [], "test": []}
            for fn in glob(join(data_absolute_path, "**/*.wav"), recursive=True):
                dic_data["relative path"].append(self.get_id(fn))

            dic_data["train"] = np.full(len(dic_data["relative path"]), False)
            dic_data["dev"] = np.full(len(dic_data["relative path"]), False)
            dic_data["test"] = np.full(len(dic_data["relative path"]), True)
            self.df_data = pd.DataFrame(data=dic_data)

        # add infos to the dataframe
        df_base = pd.read_excel(join(c2si_root_folderpath, gtpath), sheet_name="base")
        # remove lines with no id
        df_base = df_base[df_base["ID-RUGBI"].notnull()]

        nb_rows = self.df_data.shape[0]
        ids_rugbi = np.empty(nb_rows, dtype=int)
        sessions = np.empty(nb_rows, dtype=int)
        intel_scores = np.empty(nb_rows, dtype=float)
        sev_scores = np.empty(nb_rows, dtype=float)
        loc_records = np.empty(nb_rows, dtype=str)
        tasks = np.empty(nb_rows, dtype=str)
        groups = np.empty(nb_rows, dtype=float)
        sex = np.empty(nb_rows, dtype=float)
        ages = np.empty(nb_rows, dtype=float)
        for index, row in self.df_data.iterrows():
            filepath = row["relative path"]
            loc_records[index], task, _, id_rugbi, session = get_infos(filepath)
            tasks[index] = task
            ids_rugbi[index] = id_rugbi
            sessions[index] = session

            gt_row_selected = df_base.loc[(df_base['ID-RUGBI'] == id_rugbi) & (df_base["enr2x"] == session)]
            groups[index] = gt_row_selected["groupe"].values[0]
            sex[index] = gt_row_selected["sexe"].values[0]
            ages[index] = gt_row_selected["age"].values[0]
            if task == "L":
                intel_scores[index] = gt_row_selected["intellec"].values[0]
                sev_scores[index] = gt_row_selected["sevlec"].values[0]
            else:
                intel_scores[index] = gt_row_selected["intel"].values[0]
                sev_scores[index] = gt_row_selected["sev"].values[0]

        self.df_data = self.df_data.assign(id_rugbi=ids_rugbi)
        self.df_data = self.df_data.assign(session=sessions)
        self.df_data = self.df_data.assign(loc_record=loc_records)
        self.df_data = self.df_data.assign(task=tasks)
        self.df_data = self.df_data.assign(intel=intel_scores)
        self.df_data = self.df_data.assign(sev=sev_scores)
        self.df_data = self.df_data.assign(group=groups)
        self.df_data = self.df_data.assign(sex=sex)
        self.df_data = self.df_data.assign(age=ages)

        # select taks and groups
        self.select_data(targeted_tasks, group)

    def select_data(self, targeted_tasks, group):
        """Select data according to tasks and the group."""
        selected_groups = self.df_data.group.isin(GROUPS[group])
        selected_tasks = self.df_data.task.isin(targeted_tasks)
        self.df_selected = self.df_data[selected_groups & selected_tasks]

    @property
    def training_set(self):
        """Return array of filepaths from training set."""
        train_list = self.df_selected[self.df_selected["train"]]["relative path"].values
        return [filepath + ".wav" for filepath in train_list]

    @property
    def devel_set(self):
        """Return array of filepaths from development set."""
        dev_list = self.df_selected[self.df_selected["dev"]]["relative path"].values
        return [filepath + ".wav" for filepath in dev_list]

    @property
    def testing_set(self):
        """Return array of filepaths from testing set."""
        test_list = self.df_selected[self.df_selected["test"]]["relative path"].values
        return [filepath + ".wav" for filepath in test_list]

    @property
    def gt_size(self):
        """Return the size of the ground_truth."""
        i = 0
        i += 1 if self.severity_score else 0
        i += 1 if self.intelligibility_score else 0

        if self.regression_scales is not None:
            return (len(self.regression_scales), i)

        return i

    def get_samples_time_in(self, filepath):
        """Return a list of tuples corresponding to the start and end times of each sample.

        Parameters:
        -----------
        filepath: str
            Filepath of the audio file we want to get the ground truth times.
        """
        info_file = sf.info(filepath)
        return [(0, info_file.samplerate*info_file.duration)]

    def _fill_output(self, id_audio, sample_begin, sample_end, output):
        """Tool to fill an output array.

        Parameters
        ----------
        id_audio: str
            Id of the audio file, it is the relative path.

        sample_begin: integer > 0

        sample_end: integer > 0

        output: np.array
            Array to fill with ground truth (supposed zeros).
        """
        selected_row = self.df_data[self.df_data["relative path"] == id_audio]
        sev = selected_row["sev"].values
        intel = selected_row["intel"].values

        if self.regression_scales is not None:
            j = 0
            if self.severity_score:
                i = 0
                for interval in self.regression_scales:
                    if interval[0] <= sev <= interval[1]:
                        output[i, j] = 1
                        break

                    i += 1

                j = 1

            if self.intelligibility_score:
                i = 0
                for interval in self.regression_scales:
                    if interval[0] <= intel <= interval[1]:
                        output[i, j] = 1
                        break

                    i += 1

        else:
            i = 0
            if self.severity_score:
                output[0] = sev
                i = 1

            if self.intelligibility_score:
                output[i] = intel


    def get_output_description(self):
        """Return a list that describe the output."""
        output = []

        if self.regression_scales is not None:
            if self.severity_score:
                for interval in self.regression_scales:
                    output.append([interval, "severity"])

            if self.intelligibility_score:
                for interval in self.regression_scales:
                    output.append([interval, "intelligibility"])
        else:
            if self.severity_score:
                output.append("severity")
                output.append("intelligibility")


        return output

    def get_gt_for(self, filepath):
        """Get tuples corresponding to the start, end times of each sample and
        the ground truth expected.

        Parameters:
        -----------
        filepath: str
            Filepath of the audio file we want to get the ground truth.
        """
        raise Exception("Not yet implemented")

    def get_majority_gt_at_sample(self, filepath, sample_begin, sample_end):
        """Return an integer that represent the majority class for a specific sample."""
        raise Exception("Not possible with the C2SI dataset")


def get_infos(filepath):
    """Return the infos related to the id."""
    match = re.match("(.*)/(.*)-(.*)-(.*)-(.).*", filepath)
    loc_record = match[2]
    session_number = match[4]
    task = match[5]
    id_base = match[1]
    id_rugbi = match[3]

    return loc_record, task, id_base, int(id_rugbi), int(session_number)


def create_splits(c2si_root_folderpath, output_filename, datapath="16k", gtpath="base.xlsx"):
    """Create splits for C2SI corpus.

    Parameters
    ----------
    """
    # TODO
