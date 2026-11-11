#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    """Mean pairwise BPR loss used in Eq. (13)."""

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        # Stabilize the log of the pairwise preference probability.
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        loss = loss.mean()

        return loss


class EmbLoss(nn.Module):
    """Sum embedding norms normalized by their row counts."""


    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = 0
        for embedding in embeddings:
            tmp = torch.norm(embedding, p=self.norm)
            tmp = tmp / embedding.shape[0]
            emb_loss += tmp
        return emb_loss

