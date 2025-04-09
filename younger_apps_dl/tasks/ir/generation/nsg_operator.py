#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-14 18:07:43
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-08 22:50:02
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import torch
import pathlib

from typing import Literal, Callable
from pydantic import BaseModel, Field
from collections import OrderedDict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, top_k_accuracy_score

from younger_apps_dl.tasks import BaseTask, register_task
from younger_apps_dl.engines import StandardTrainer, StandardTrainerOptions, StandardEvaluator, StandardEvaluatorOptions, GraphSplit, GraphSplitOptions
from younger_apps_dl.datasets import GraphDataset, GraphData
from younger_apps_dl.models import MAEGIN


class ModelOptions(BaseModel):
    model_type: Literal['MAEGIN'] = Field('MAEGIN', description='The identifier of the model type, e.g., \'MAEGIN\', etc.')
    node_emb_dim: int = Field(512, description='Node embedding dimensionality.')
    hidden_dim: int = Field(256, description='Hidden layer dimensionality within the model.')
    dropout_rate: float = Field(0.5, description='Dropout probability used for regularization.')
    layer_number: int = Field(3, description='Number of layers (e.g., message-passing rounds for GNNs).')


class OptimizerOptions(BaseModel):
    lr: float = Field(0.001, description='Learning rate used by the optimizer.')
    eps: float = Field(1e-8, description='Epsilon for numerical stability.')
    weight_decay: float = Field(0.01, description='L2 regularization (weight decay) coefficient.')
    amsgrad: bool = Field(False, description='Whether to use the AMSGrad variant of the Adam optimizer.')


class SchedulerOptions(BaseModel):
    start_factor: float = Field(0.1, description='Initial learning rate multiplier for warm-up.')
    warmup_steps: int = Field(1500, description='Number of warm-up steps at the start of training.')
    total_steps: int = Field(150000, description='Total number of training steps for the scheduler to plan the learning rate schedule.')
    last_step: int = Field(-1, description='The last step index when resuming training. Use -1 to start fresh.')


class DatasetOptions(BaseModel):
    meta_filepath: pathlib.Path = Field(..., description='Path to the metadata file that describes the dataset.')
    raw_dirpath: pathlib.Path = Field(..., description='Directory containing raw input data files.')
    processed_dirpath: pathlib.Path = Field(..., description='Directory where processed dataset should be stored.')
    worker_number: int = Field(4, description='Number of workers for parallel data loading or processing.')


class GraphEmbeddingOptions(BaseModel):
    # Main Options
    logging_filepath: pathlib.Path | None = Field(None, description='Logging file path where logs will be saved, default to None, which may save to a default path that is determined by the Younger.')

    mask_ratio: float = Field(..., description='')
    mask_method: Literal['Random', 'Purpose'] = Field(..., description='')

    trainer: StandardTrainerOptions
    evaluator: StandardEvaluatorOptions
    preprocessor: GraphSplitOptions

    train_dataset: DatasetOptions
    valid_dataset: DatasetOptions
    test_dataset: DatasetOptions

    model: ModelOptions
    optimizer: OptimizerOptions
    scheduler: SchedulerOptions


# Self-Supervised Learning for Node Prediction
@register_task('ir', 'graph_embedding')
class GraphEmbedding(BaseTask[GraphEmbeddingOptions]):
    OPTIONS = GraphEmbeddingOptions
    def train(self):
        self.train_dataset = self._build_dataset_(
            self.options.train_dataset.meta_filepath,
            self.options.train_dataset.raw_dirpath,
            self.options.train_dataset.processed_dirpath,
            'train',
            self.options.train_dataset.worker_number
        )
        self.valid_dataset = self._build_dataset_(
            self.options.valid_dataset.meta_filepath,
            self.options.valid_dataset.raw_dirpath,
            self.options.valid_dataset.processed_dirpath,
            'valid',
            self.options.valid_dataset.processed_dirpath
        )
        self.model = self._build_model_(
            len(self.train_dataset.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            self.options.model.dropout_rate,
            self.options.model.layer_number
        )
        self.optimizer = self._build_optimizer_(
            self.model,
            self.options.optimizer.lr,
            self.options.optimizer.eps,
            self.options.optimizer.weight_decay,
            self.options.optimizer.amsgrad,
        )
        self.scheduler = self._build_scheduler_(
            self.optimizer,
            self.options.scheduler.start_factor,
            self.options.scheduler.warmup_steps,
            self.options.scheduler.total_steps,
            self.options.scheduler.last_step,
        )
        self.dicts = self.train_dataset.dicts

        trainer = StandardTrainer(self.options.trainer)
        trainer.run(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_dataset,
            self.valid_dataset,
            self._train_fn_,
            self._valid_fn_,
            self._on_step_end_fn_,
            self._on_epoch_end_fn_,
            'pyg',
            self.options.logging_filepath
        )

    def evaluate(self):
        self.test_dataset = self._build_dataset_(
            self.options.test_dataset.meta_filepath,
            self.options.test_dataset.raw_dirpath,
            self.options.test_dataset.processed_dirpath,
            'test',
            self.options.test_dataset.processed_dirpath
        )
        self.model = self._build_model_(
            len(self.test_dataset.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            self.options.model.dropout_rate,
            self.options.model.layer_number
        )
        self.dicts = self.test_dataset.dicts

        evaluator = StandardEvaluator(self.options.evaluator)
        evaluator.run(
            self.model,
            self.test_dataset,
            self._test_fn_,
            'pyg',
            self.options.logging_filepath
        )

    def preprocess(self):
        preprocessor = GraphSplit(self.options.preprocessor)
        preprocessor.run(self.options.logging_filepath)

    def _build_model_(self, node_emb_size: int, node_emb_dim: int, hidden_dim: int, dropout_rate: float, layer_number: int) -> torch.nn.Module:
        model = MAEGIN(
            node_emb_size,
            node_emb_dim,
            hidden_dim,
            dropout_rate,
            layer_number
        )
        return model

    def _build_dataset_(self, meta_filepath: pathlib.Path, raw_dirpath: pathlib.Path, processed_dirpath: pathlib.Path, split: Literal['train', 'valid', 'test'], worker_number: int) -> GraphDataset:
        dataset = GraphDataset(
            meta_filepath,
            raw_dirpath,
            processed_dirpath,
            split=split,
            worker_number=worker_number
        )
        return dataset

    def _build_optimizer_(
        self,
        model: torch.nn.Module,
        lr: float,
        eps: float,
        weight_decay: float,
        amsgrad: float
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        return optimizer

    def _build_scheduler_(
        self,
        optimizer: torch.nn.Module,
        start_factor: float,
        warmup_steps: int,
        total_steps: int,
        last_step: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        warmup_lr_schr = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            total_iters=warmup_steps,
            last_epoch=last_step,
        )
        cosine_lr_schr = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            last_epoch=last_step,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_schr, cosine_lr_schr],
            milestones=[warmup_steps],
            last_epoch=last_step,
        )
        return scheduler

    def _train_fn_(self, model: torch.nn.Module, minibatch: GraphData) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        device_descriptor = next(model.parameters()).device
        minibatch = minibatch.to(device_descriptor)
        x, edge_index, golden = self.mask(minibatch, self.dicts['t2i'], self.options.mask_ratio, self.options.mask_method)

        output = self.model(x, edge_index)

        loss = torch.nn.functional.cross_entropy(output, golden.squeeze(1), ignore_index=-1)

        train_metric_names = ['loss']
        train_metric_values = [loss]
        train_metric_formats = [lambda x: f'{x:.4f}']
        return train_metric_names, train_metric_values, train_metric_formats

    def _valid_fn_(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        device_descriptor = next(model.parameters()).device

        outputs = list()
        goldens = list()
        # Return Output & Golden
        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader)) as progress_bar:
                for index, minibatch in enumerate(dataloader, start=1):
                    minibatch = minibatch.to(device_descriptor)
                    x, edge_index, golden = self.mask(minibatch, self.dicts['t2i'], self.options.mask_ratio, self.options.mask_method)
                    output = torch.softmax(self.model(x, edge_index), dim=-1)

                    outputs.append(output)
                    goldens.append(golden)
                    progress_bar.update(1)

        outputs = torch.cat(outputs)
        goldens = torch.cat(goldens).squeeze()

        val_indices = goldens != -1
        outputs = outputs[val_indices]
        goldens = goldens[val_indices]

        score = outputs.cpu().numpy()
        pred = outputs.max(1)[1].cpu().numpy()
        gold = goldens.cpu().numpy()

        print("pred[:5]:", pred[:5])
        print("gold[:5]:", gold[:5])

        acc = accuracy_score(gold, pred)
        macro_p = precision_score(gold, pred, average='macro', zero_division=0)
        macro_r = recall_score(gold, pred, average='macro', zero_division=0)
        macro_f1 = f1_score(gold, pred, average='macro', zero_division=0)
        micro_f1 = f1_score(gold, pred, average='micro', zero_division=0)
        top3_acc = top_k_accuracy_score(gold, score, k = 3, labels=range(score.shape[1]))
        top5_acc = top_k_accuracy_score(gold, score, k = 5, labels=range(score.shape[1]))

        metrics = [
            ('acc', acc, lambda x: f'{x:.4f}'),
            ('macro_p', macro_p, lambda x: f'{x:.4f}'),
            ('macro_r', macro_r, lambda x: f'{x:.4f}'),
            ('macro_f1', macro_f1, lambda x: f'{x:.4f}'),
            ('micro_f1', micro_f1, lambda x: f'{x:.4f}'),
            ('top3_acc', top3_acc, lambda x: f'{x:.4f}'),
            ('top5_acc', top5_acc, lambda x: f'{x:.4f}') 
        ]
        valid_metric_names = list()
        valid_metric_values = list()
        valid_metric_formats = list()
        for metric_name, metric_value, metric_format in metrics:
            valid_metric_names.append(metric_name)
            valid_metric_values.append(metric_value)
            valid_metric_formats.append(metric_format)

        return valid_metric_names, valid_metric_values, valid_metric_formats

    def _test_fn_(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        return self._valid_fn_(model, dataloader)

    def _on_step_end_fn_(self, step: int) -> None:
        self.scheduler.step()
        return

    def _on_epoch_end_fn_(self, epoch: int) -> None:
        return

    def mask(self, minibatch: GraphData, dict: dict[str, int], mask_ratio: float, mask_method: Literal['Random', 'Purpose']) -> tuple[torch.Tensor, torch.Tensor]:
        x = minibatch.x
        edge_index = minibatch.edge_index

        label = x.clone()
        source_index = edge_index[0].to(self.device_descriptor)
        target_index = edge_index[1].to(self.device_descriptor)
        
        node_without_successor = target_index[~torch.isin(target_index, source_index)]

        unique = torch.unique(node_without_successor)
        if unique.shape[0] != 0:
            mask_ratio = mask_ratio * x.shape[0] / unique.shape[0]
            mask_ratio = 1 if mask_ratio > 1 else mask_ratio

        mask_probability = torch.zeros_like(x, dtype=torch.float, device=self.device_descriptor)
        mask_probability[node_without_successor] = torch.full(x.shape, mask_ratio, dtype=torch.float, device=self.device_descriptor)[node_without_successor]
        self.mask_indices = torch.bernoulli(mask_probability).to(self.device_descriptor).bool()
        label[~self.mask_indices] = -1
        
        if not is_test:
            self.mask_mask_indices = torch.bernoulli(
                torch.full(x.shape, 0.8, dtype=torch.float, device=self.device_descriptor)).bool() & self.mask_indices
            x[self.mask_mask_indices] = x_dict['__MASK__']

            self.mask_optr_indices = torch.bernoulli(torch.full(x.shape, 0.5, dtype=torch.float,
                                                                device=self.device_descriptor)).bool() & self.mask_indices & ~self.mask_mask_indices
            x[self.mask_optr_indices] = \
            torch.randint(2, len(x_dict), x.shape, dtype=torch.long, device=self.device_descriptor)[self.mask_optr_indices]
        else:

            self.mask_mask_indices = torch.bernoulli(
                torch.full(x.shape, 1, dtype=torch.float, device=self.device_descriptor)).bool() & self.mask_indices
            x[self.mask_mask_indices] = x_dict['__MASK__']

        label = x.clone()
        mask_probability = torch.full(x.shape, mask_ratio, dtype=torch.float, device=self.device_descriptor)
        mask_indices = torch.bernoulli(mask_probability).to(self.device_descriptor).bool()
        label[~mask_indices] = -1

        mask_mask_indices = torch.bernoulli(torch.full(x.shape, 0.8, dtype=torch.float, device=self.device_descriptor)).bool() & mask_indices
        x[mask_mask_indices] = x_dict['__MASK__']

        mask_optr_indices = torch.bernoulli(torch.full(x.shape, 0.5, dtype=torch.float, device=self.device_descriptor)).bool() & mask_indices & ~mask_mask_indices
        x[mask_optr_indices] = torch.randint(2, len(x_dict), x.shape, dtype=torch.long, device=self.device_descriptor)[mask_optr_indices]

        label = x.clone()
        mask_probability = torch.full(x.shape, mask_ratio, dtype=torch.float, device=self.device_descriptor)
        self.mask_indices = torch.bernoulli(mask_probability).to(self.device_descriptor).bool()
        label[~self.mask_indices] = -1

        if not is_test:
            self.mask_mask_indices = torch.bernoulli(
                torch.full(x.shape, 0.8, dtype=torch.float, device=self.device_descriptor)).bool() & self.mask_indices
            x[self.mask_mask_indices] = x_dict['__MASK__']

            self.mask_optr_indices = torch.bernoulli(torch.full(x.shape, 0.5, dtype=torch.float,
                                                                device=self.device_descriptor)).bool() & self.mask_indices & ~self.mask_mask_indices
            x[self.mask_optr_indices] = \
            torch.randint(2, len(x_dict), x.shape, dtype=torch.long, device=self.device_descriptor)[self.mask_optr_indices]
        else:
            self.mask_mask_indices = torch.bernoulli(
                torch.full(x.shape, 1, dtype=torch.float, device=self.device_descriptor)).bool() & self.mask_indices
            x[self.mask_mask_indices] = x_dict['__MASK__']

        return x, label
    
        return x, label

