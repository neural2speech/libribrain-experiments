import numpy as np
import torch


class PredictionSmootherDataset(torch.utils.data.Dataset):
    """
    Turn raw model probabilities + ground-truth labels into (B, 1, T) windows
    to train a sequence smoother.

    Expected CSV format (as produced by make_submission.py):
      - preds_csv:  2 columns -> [segment_idx, speech_prob]
      - labels_csv: 2 columns -> [segment_idx, speech_label]

    Parameters
    ----------
    preds_csv : str
        Path to CSV with model probabilities.
    labels_csv : str
        Path to CSV with ground-truth labels (train/val/test splits only).
    seq_len : int
        Window length in samples.
    stride : int
        Sliding-window hop (1 = every sample).
    pad : bool
        If True, pad both sides with edge values so that we also build
        windows around the first and last samples.
    """

    def __init__(
        self,
        preds_csv: str,
        labels_csv: str,
        seq_len: int = 200,
        stride: int = 1,
        pad: bool = True,
        **_ignored,
    ):

        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.pad = bool(pad)

        # load CSVs
        # (skip header, keep 2nd column)
        probs = np.loadtxt(preds_csv, delimiter=",", skiprows=1, usecols=1)
        labels = np.loadtxt(labels_csv, delimiter=",", skiprows=1, usecols=1).astype(
            np.int64
        )

        assert (
            probs.shape[0] == labels.shape[0]
        ), "preds and labels must have the same length"

        if pad:
            pad_w = self.seq_len // 2
            probs = np.pad(probs, (pad_w, pad_w), mode="edge")
            labels = np.pad(labels, (pad_w, pad_w), mode="edge")

        self.probs = torch.from_numpy(probs.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

        self.idxs = list(range(0, len(self.labels) - self.seq_len + 1, self.stride))

        # Bookkeeping so the rest of the code keeps working
        self.labels_sorted = [0, 1]
        self.channel_means = None
        self.channel_stds = None

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        start = self.idxs[i]
        end = start + self.seq_len
        x = self.probs[start:end].unsqueeze(0)  # (1, T)
        y = self.labels[start:end]  # (T,)
        return x, y
