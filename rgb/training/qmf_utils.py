"""
Quality-aware Multimodal Fusion (QMF) utilities.

Ported from: https://github.com/QingyangZhang/QMF
Paper: "Provably Dynamic Multimodal Fusion" (Zhang & Wu, ICML 2023)

Contains:
  - History: tracks per-sample cumulative loss for confidence calibration
  - rank_loss: MarginRankingLoss that enforces confident samples rank higher
"""

import numpy as np
import torch
import torch.nn as nn


class History:
    """Tracks per-sample cumulative loss across epochs.

    Samples that consistently have low loss (i.e., the model is correct
    and confident on them) accumulate high correctness scores.
    The rank loss then enforces that such samples also have high
    energy-based confidence scores.
    """

    def __init__(self, n_data):
        self.correctness = np.zeros(n_data)
        self.confidence = np.zeros(n_data)

    def correctness_update(self, data_idx, loss_values, confidence):
        """Update correctness history with per-sample loss values.

        Args:
            data_idx: tensor of dataset indices for this batch
            loss_values: per-sample loss (detached, no grad)
            confidence: per-sample confidence scores (detached)
        """
        idx = data_idx.cpu().numpy()
        self.correctness[idx] += loss_values.cpu().numpy()
        self.confidence[idx] = confidence.cpu().detach().numpy()

    def correctness_normalize(self, data):
        """Normalize correctness to [0, 1] range."""
        data_min = self.correctness.min()
        data_max = float(self.correctness.max())
        if data_max - data_min < 1e-8:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)

    def get_target_margin(self, data_idx1, data_idx2):
        """Get ranking target and margin for a pair of samples.

        Returns:
            target: +1 if idx1 should rank higher, -1 if lower, 0 if equal
            margin: absolute difference in normalized correctness
        """
        idx1 = data_idx1.cpu().numpy()
        idx2 = data_idx2.cpu().numpy()

        cum1 = self.correctness_normalize(self.correctness[idx1])
        cum2 = self.correctness_normalize(self.correctness[idx2])

        greater = (cum1 > cum2).astype(np.float32)
        less = (cum1 < cum2).astype(np.float32) * (-1)
        target = torch.from_numpy(greater + less).float()

        margin = torch.from_numpy(np.abs(cum1 - cum2)).float()

        return target, margin


def rank_loss(confidence, idx, history):
    """Compute ranking loss to calibrate confidence scores.

    Pairs each sample with the next sample in the batch (circular roll),
    and enforces that samples with higher cumulative correctness have
    higher confidence scores.

    Args:
        confidence: (B,) energy-based confidence scores
        idx: (B,) dataset indices for this batch
        history: History object tracking cumulative losses
    """
    confidence = confidence.squeeze()
    assert confidence.dim() == 1

    # Pair each sample with the next (circular)
    rank_input1 = confidence
    rank_input2 = torch.roll(confidence, -1)
    idx2 = torch.roll(idx, -1)

    # Get target ordering and margin from history
    rank_target, rank_margin = history.get_target_margin(idx, idx2)
    rank_target = rank_target.to(confidence.device)
    rank_margin = rank_margin.to(confidence.device)

    # Add margin to the second input
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1
    rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

    return nn.MarginRankingLoss(margin=0.0)(
        rank_input1, rank_input2, -rank_target)
