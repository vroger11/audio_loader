"""Module to get dataloaders that fill the CPU RAM."""
import torch

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


def get_dataset_fixed_size(sampler, selected_set):
    """Create dataset that fill the CPU RAM with segments of the same size.

    Parameters:
    -----------
    sampler: Sampler
        audio_loader sampler that sample segments with the same size.

    selected_set: str
       can be "train" or "training" for the training set
              'valid', 'validation', 'dev' or "devel" for the validation set
              'test' or "testing" for the test set.

    Return:
    -------
    Dataset designed for fixed segments size.
    """
    return DatasetFromArray(list(sampler.get_samples_from(selected_set)))


def get_dataloader_fixed_size(sampler, batch_size, selected_set):
    """Create dataloader that fill the CPU RAM with segments of the same size.

    Parameters:
    -----------
    sampler: Sampler
        audio_loader sampler that sample segments with the same size.

    batch_size: int
        Number of samples per batch.

    selected_set: str
       can be "train" or "training" for the training set
              'valid', 'validation', 'dev' or "devel" for the validation set
              'test' or "testing" for the test set.

    Return:
    -------
    Dataloader designed for fixed segments size.
    """
    train_set = selected_set in ["train", "training"]
    return DataLoader(get_dataset_fixed_size(sampler, selected_set),
                      batch_size=batch_size, shuffle=train_set, drop_last=train_set)


def get_dataset_dynamic_size(sampler, selected_set):
    """Create dataset that fill the CPU RAM with segments of different sizes.

    Parameters:
    -----------
    sampler: Sampler
        audio_loader sampler that samples segments with the same size.

    selected_set: str
       can be "train" or "training" for the training set
              'valid', 'validation', 'dev' or "devel" for the validation set
              'test' or "testing" for the test set.

    Return:
    -------
    Dataloader with padded samples designed for RNN. The ground truth is not
    padded. Each batch return packed padded sequences.
    """
    # multichannel not supported for pytorch
    # TODO find a solution for multichannels
    data = [(torch.tensor(x.reshape(*x.shape[1:])), torch.tensor(y)) for x, y in sampler.get_samples_from(selected_set)]
    dataset = DatasetFromArray(list(data))

    if sampler.supervised:
        pad_collate_func = pad_collate_supervised
    else:
        pad_collate_func = pad_collate_unsupervised

    return dataset, pad_collate_func


def get_dataloader_dynamic_size(sampler, batch_size, selected_set):
    """Create dataloader that fill the CPU RAM with segments of different sizes.

    Parameters:
    -----------
    sampler: Sampler
        audio_loader sampler that samples segments with the same size.

    batch_size: int
        Number of samples per batch.

    selected_set: str
       can be "train" or "training" for the training set
              'valid', 'validation', 'dev' or "devel" for the validation set
              'test' or "testing" for the test set.

    Return:
    -------
    Dataloader with padded samples designed for RNN. The ground truth is not
    padded. Each batch return packed padded sequences.
    """
    train_set = selected_set in ["train", "training"]
    dataset, pad_collate_func = get_dataset_dynamic_size(sampler, selected_set)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=train_set, drop_last=train_set,
                      collate_fn=pad_collate_func)


def pad_collate_supervised(batch):
    """Collate function to pad data of supervised batch."""
    (data, target) = zip(*batch)
    data_lens = [len(x) for x in data]
    data_padded = pad_sequence(data, batch_first=True, padding_value=0.)
    packed_data = pack_padded_sequence(data_padded, data_lens,
                                       batch_first=True, enforce_sorted=False)
    return packed_data, target


def pad_collate_unsupervised(batch):
    """Collate function to pad data of unsupervised batch."""
    data = zip(*batch)
    data_lens = [len(x) for x in data]
    data_padded = pad_sequence(data, batch_first=True, padding_value=0.)
    packed_data = pack_padded_sequence(data_padded, data_lens,
                                       batch_first=True, enforce_sorted=False)
    return packed_data


class DatasetFromArray(Dataset):
    """Create pytorch Dataset from a list of arrays."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

