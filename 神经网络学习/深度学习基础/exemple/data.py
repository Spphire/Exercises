from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
import os



class MNIST(Dataset):
    def __init__(self, path, mode):
        with open(os.path.join(path, f"{mode}-images-idx3-ubyte"), "rb") as fp:
            magic, num, rows, cols = struct.unpack('>IIII', fp.read(16))
            self.xs = np.fromfile(fp, dtype=np.uint8).reshape(-1, 1, 28, 28).astype(np.float32)
        with open(os.path.join(path, f"{mode}-labels-idx1-ubyte"), "rb") as fp:
            magic, num = struct.unpack('>II', fp.read(8))
            self.ys = np.fromfile(fp, dtype=np.uint8).astype(np.long)

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, item):
        return self.xs[item] / 255., self.ys[item]
        # [1x28x28] numpy            [] 
        # batch=128
        # stack([]) -> [128x1x28x28] tensor
        # stack()   -> [128] tensor

def create_train_loader(batch_size):
    data = MNIST(path="MNIST", mode="train")
    loader = DataLoader(data, batch_size, shuffle=True)
    return loader


def create_test_loader(batch_size):
    data = MNIST(path="MNIST", mode="t10k")
    loader = DataLoader(data, batch_size)
    return loader
