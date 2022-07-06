"""Knowledge Graph embedding model optimizers."""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

from torch import nn


class ECROptimizer(object):

    def __init__(self, model, optimizer, valid_freq, batch_size,
                 regularizer=None, use_cuda=False, dropout=0.,):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.valid_freq = valid_freq
        self.dropout = dropout

    def reduce_lr(self, factor=0.8):
        """Reduce learning rate.

        Args:
            factor: float for the learning rate decay
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor

    def calculate_loss(self, examples):
        pass

        predictions, factors = self.model(examples)    # list(所有可能实体),行
        loss = self.loss_fn(predictions)
        if self.regularizer:
            loss += self.regularizer.forward(factors)
        return loss

    def epoch(self, train_set):

        losses = []
        idx = [_ for _ in range(len(train_set))]
        random.shuffle(idx)
        for train_sample_num in tqdm(idx):
            pass

            loss = self.calculate_loss(train_set[train_sample_num])

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(loss.item())

        return np.mean(losses)

    def evaluate(self, test_data, valid_losses=False):
        pass

        valid_losses = []
        valid_loss = None

        with torch.no_grad():
            for idx in enumerate(tqdm(len(test_data))):

                loss = self.calculate_loss(test_data[idx])
                valid_losses.append(loss.item())

            if valid_losses:
                valid_loss = np.mean(valid_losses)

            return valid_loss
