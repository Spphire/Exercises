from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
import os



class MNIST(Dataset):
    def __init__(self, path, mode):

        if mode=="test" :
            with open(os.path.join(path, f"{mode}_batch"), "rb") as fp:
                self.xs = np.fromfile(fp, dtype=np.uint8).reshape(-1, 3 * 32 * 32 + 1).astype(np.float32)[ : , 1::1].reshape(-1 , 3 , 32 , 32)
            with open(os.path.join(path, f"{mode}_batch"), "rb") as fp:
                self.ys = np.fromfile(fp, dtype=np.uint8).reshape(-1, 3 * 32 * 32 + 1).astype(np.float32)[ : , 0].reshape(-1)
            
        if mode=="train" :

            with open(os.path.join(path, f"{mode}_batch_1"), "rb") as fp:
                self.xs = np.fromfile(fp, dtype=np.uint8).reshape(-1, 3 * 32 * 32 + 1).astype(np.float32)[:,1::1].reshape(-1, 3, 32, 32)
            with open(os.path.join(path, f"{mode}_batch_1"), "rb") as fp:
                self.ys = np.fromfile(fp, dtype=np.uint8).reshape(-1, 3 * 32 * 32 + 1).astype(np.float32)[:,0].reshape(-1)

            
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
    data = MNIST(path="MNIST", mode="test")
    loader = DataLoader(data, batch_size)
    return loader
