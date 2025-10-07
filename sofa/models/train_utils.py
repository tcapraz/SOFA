#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset

class MuDataDataset(Dataset):
    def __init__(self, X=None, Y=None, Xmask=None, Ymask=None, device="cpu"):
        """
        Args:
            X (list | None): list of modality matrices for inputs
            Y (list | None): list of modality matrices for outputs
            Xmask (list | None): list of boolean vectors (n_samples,)
            Ymask (list | None): list of boolean vectors (n_samples,)
            device (str): 'cpu' or 'cuda'
        """
        self.X = X
        self.Y = Y
        self.Xmask = Xmask
        self.Ymask = Ymask
        self.device = device

        # infer number of samples
        if X is not None and len(X) > 0:
            self.n_obs = X[0].shape[0]
        elif Y is not None and len(Y) > 0:
            self.n_obs = Y[0].shape[0]
        else:
            self.n_obs = 1 # arbitrary number to allow empty model instances

       # consistency checks
        if X is not None:
            assert all(x.shape[0] == self.n_obs for x in X), "All X modalities must have same n_obs"
        if Y is not None:
            assert all(y.shape[0] == self.n_obs for y in Y), "All Y modalities must have same n_obs"
        if Xmask is not None:
            assert len(Xmask) == len(X), "Xmask must match number of X modalities"
            for xm in Xmask:
                assert xm.shape == (self.n_obs,), f"Xmask should be 1D of length {self.n_obs}"
        if Ymask is not None:
            assert len(Ymask) == len(Y), "Ymask must match number of Y modalities"
            for ym in Ymask:
                assert ym.shape == (self.n_obs,), f"Ymask should be 1D of length {self.n_obs}"

        self.indices = list(range(self.n_obs))

    def __len__(self):
        return self.n_obs

    def __getitem__(self, idx):
        sample = {"X": None, "Y": None, "Xmask": None, "Ymask": None, "idx": idx}

        # --- X modalities ---
        if self.X is not None:
            sample["X"] = []
            for mod in self.X:
                x = mod[idx]
                if hasattr(x, "toarray"):
                    x = x.toarray().squeeze()
                sample["X"].append(torch.as_tensor(x, dtype=torch.float32))

        # --- Y modalities ---
        if self.Y is not None:
            sample["Y"] = []
            for mod in self.Y:
                y = mod[idx]
                if hasattr(y, "toarray"):
                    y = y.toarray().squeeze()
                sample["Y"].append(torch.as_tensor(y, dtype=torch.float32))

        # --- X masks (scalar per modality per sample) ---
        if self.Xmask is not None:
            sample["Xmask"] = [
                torch.as_tensor(self.Xmask[i][idx], dtype=torch.bool)
                for i in range(len(self.Xmask))
            ]

        # --- Y masks (scalar per modality per sample) ---
        if self.Ymask is not None:
            sample["Ymask"] = [
                torch.as_tensor(self.Ymask[i][idx], dtype=torch.bool)
                for i in range(len(self.Ymask))
            ]

        return sample
