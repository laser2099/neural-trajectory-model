from torch.utils.data import Dataset
import numpy as np

class SDFDataset(Dataset):
    def __init__(self, npy_path, subsets=None) -> None:
        self.path = npy_path
        data = np.load(npy_path, allow_pickle=True)[()]
        self.subsets = subsets or [k for k in data.keys() if "_sdf" not in k]
        self.points = np.concatenate([data[k] for k in self.subsets])
        self.sdfs = np.concatenate([data[k + "_sdf"] for k in self.subsets])

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index) -> dict:
        return self.points[index], self.sdfs[index]
