import torch
import torch.nn as nn


class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight = 1.0):
        """
        Binary Cross-Entropy Loss with Deterministic Label Smoothing.

        Parameters:
            smoothing (float): Smoothing factor. Must be between 0 and 1.
            pos_weight (float): Weight for the positive class.
        """
        super().__init__()
        self.smoothing = float(smoothing)

        # make sure pos_weight is a 1-element 1-D tensor (shape = [1])
        if not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor([pos_weight], dtype=torch.float)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, target):
        """
        Parameters:
            logits (Tensor, shape (B, 1) or (B,))
            target (Tensor | list | ndarray | float):
                Binary labels (0/1).  Will be broadcast into `(B, 1)`.

        Returns:
            Tensor: Smoothed BCE-with-logits loss.
        """
        # convert anything that is not yet a tensor
        if not torch.is_tensor(target):
            target = torch.as_tensor(target, device=logits.device)

        # logits come out as (B,1); make target match that shape
        if target.dim() == 1:
            target = target.unsqueeze(1)

        target = target.float()  # Ensure target is a float tensor
        target_smoothed = target * (1 - self.smoothing) + self.smoothing * 0.5
        return self.bce_loss(logits, target_smoothed)
