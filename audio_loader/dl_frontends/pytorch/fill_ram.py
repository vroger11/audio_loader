"""Module to get a sampler that fill the CPU RAM."""
from torch.utils.data import DataLoader, Dataset


def get_pytorch_dataloader_fill_ram(sampler_iterator, batch_size, selected_set):
    """Create dataloader that fill the CPU RAM."""

    class DatasetFromArray(Dataset):
        """Create pytorch Dataset from arrays."""
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    train_set = selected_set in ["train", "training"]
    dataset = DatasetFromArray(list(sampler_iterator.get_samples_from(selected_set)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train_set, drop_last=train_set)
