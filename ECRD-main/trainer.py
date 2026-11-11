#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import tempfile

import numpy as np
import torch
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_set import DataSet
from metrics import metrics_dict


class Trainer:
    def __init__(self, model, dataset: DataSet, args):
        self.model = model.to(args.device)
        self.dataset = dataset
        self.behaviors = args.behaviors
        self.topk = tuple(args.topk)
        self.metrics = tuple(metric.lower() for metric in args.metrics)
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.model_name = args.model_name
        self.check_point = getattr(args, 'check_point', args.model_name + '.pth')
        self.device = args.device
        self.delta_e = 5
        self.best_epoch = None
        self.best_validation_ndcg = None
        self.optimizer = self.get_optimizer(self.model)

    def get_optimizer(self, model):
        # Eq. (13) includes L2 in the model loss.
        return optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=self.learning_rate,
        )

    def train_model(self):
        train_data = self.dataset.behavior_dataset()
        if not len(train_data) or self.epochs < 1:
            raise ValueError('Training requires a nonempty dataset and at least one epoch.')
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        validation_data = self.dataset.validate_dataset()
        best_state = None
        self.best_epoch = None
        self.best_validation_ndcg = -float('inf')

        for epoch in range(self.epochs):
            self.model.train()
            # Sec. V-A4: Refresh the training candidate snapshot every delta_e epochs.
            if epoch % self.delta_e == 0:
                self.model.lazy_refresh_semantic_pool()
            self._train_one_epoch(train_loader, epoch)

            validation = self.evaluate(
                epoch, self.test_batch_size, validation_data,
                self.dataset.validation_interacts, self.dataset.validation_gt_length,
                include_selection_metric=True,
            )
            ndcg = validation['ndcg@10']
            logger.info('Epoch {} validation metrics: {}', epoch + 1, validation)
            # Sec. V-A3: Select the checkpoint by unrounded validation NDCG@10.
            if ndcg > self.best_validation_ndcg:
                self.best_validation_ndcg = ndcg
                self.best_epoch = epoch + 1
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                }

        if best_state is None:
            raise RuntimeError('No validation checkpoint could be selected.')
        self.model.load_state_dict(best_state)
        self.model.eval()
        reasoning = self.model.dual_cf_reasoning
        reasoning.cached_base_candidates = None
        reasoning.cached_out_candidates = None
        self.save_model(self.model)
        logger.info('Selected epoch {} with validation NDCG@10 {:.6f}',
                    self.best_epoch, self.best_validation_ndcg)

    def _train_one_epoch(self, behavior_dataset_loader, epoch):
        batches = tqdm(behavior_dataset_loader, desc=f'Train {epoch + 1}')
        total_loss = 0.0
        for batch_data in batches:
            if isinstance(batch_data, (list, tuple)):
                batch_data = batch_data[0]
            batch_ids = batch_data.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model(batch_ids)
            loss.backward()
            self.optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss
            batches.set_postfix(loss=batch_loss)

        logger.info('Epoch {} training loss {:.4f}',
                    epoch + 1, total_loss / len(behavior_dataset_loader))

    @torch.no_grad()
    def evaluate(self, epoch, test_batch_size, dataset, gt_interacts, gt_length,
                 *, mask_validation=None, include_selection_metric=False):
        if mask_validation is None:
            mask_validation = gt_interacts is self.dataset.test_interacts
        previous_mode = self.model.training
        reasoning = self.model.dual_cf_reasoning
        train_cache = reasoning.cached_base_candidates, reasoning.cached_out_candidates
        self.model.eval()
        try:
            # Refresh from the current checkpoint for both validation and testing.
            self.model.lazy_refresh_semantic_pool()
            rankings = self._ranking_hits(dataset, gt_interacts, test_batch_size,
                                          mask_validation, include_selection_metric)
            return self.calculate_result(rankings, gt_length,
                                         include_selection_metric=include_selection_metric)
        finally:
            reasoning.cached_base_candidates, reasoning.cached_out_candidates = train_cache
            self.model.train(previous_mode)

    def _ranking_hits(self, dataset, gt_interacts, batch_size, mask_validation,
                      include_selection_metric):
        if not len(dataset):
            raise ValueError('Evaluation requires a nonempty split.')
        cutoff = max(self.topk)
        if include_selection_metric:
            cutoff = max(cutoff, 10)
        cutoff = min(cutoff, self.dataset.item_count)
        if cutoff < 1:
            raise ValueError('Evaluation requires a nonempty item catalog and positive cutoffs.')

        train_items = self.dataset.train_behavior_dict[self.behaviors[-1]]
        validation_items = self.dataset.validation_interacts if mask_validation else {}
        rankings = []
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for users in tqdm(batches, desc='Evaluate'):
            user_keys = [str(user) for user in users.tolist()]
            scores = self.model.full_predict(users.to(self.device))

            # Sec. V-A3: Mask padding and known purchases before ranking.
            scores[:, 0] = -float('inf')
            for row, user in enumerate(user_keys):
                known = set(train_items.get(user, ())) | set(validation_items.get(user, ()))
                if known:
                    scores[row, list(known)] = -float('inf')

            values, top_items = torch.topk(scores, cutoff, dim=-1)
            ranked_items = top_items.cpu().numpy()
            valid_positions = torch.isfinite(values).cpu().numpy()
            for row, user in enumerate(user_keys):
                hit = np.isin(ranked_items[row], gt_interacts[user])
                rankings.append(hit & valid_positions[row])

        return np.asarray(rankings)

    def calculate_result(self, topk_list, gt_len, *, include_selection_metric=False):
        gt_len = np.asarray(gt_len)
        if len(gt_len) != len(topk_list) or np.any(gt_len <= 0):
            raise ValueError('Each evaluation user must have aligned, nonempty ground truth.')
        metrics = set(self.metrics)
        if include_selection_metric:
            metrics.add('ndcg')
        values = {
            metric: metrics_dict[metric](topk_list, gt_len).mean(axis=0)
            for metric in metrics
        }
        width = topk_list.shape[1]
        results = {
            f'{metric}@{cutoff}': float(values[metric][min(cutoff, width) - 1])
            for cutoff in self.topk for metric in self.metrics
        }
        if include_selection_metric:
            results['ndcg@10'] = float(values['ndcg'][min(10, width) - 1])
        return results

    def save_model(self, model):
        path = Path(self.model_path) / self.check_point
        path.parent.mkdir(parents=True, exist_ok=True)
        file = tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False)
        temporary = Path(file.name)
        try:
            with file:
                torch.save(model.state_dict(), file)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
