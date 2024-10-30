from torch import Tensor
from torch.utils.data import Dataset

class IndexedDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataname = Tensor([0] * len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, label = self.dataset[index]
        return sample, label, self.dataname, index