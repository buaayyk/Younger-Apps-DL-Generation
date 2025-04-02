#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-16 22:58:32
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-02 23:06:16
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import time
import torch
import pathlib

from torch import distributed
from typing import Any, Literal, Callable
from pydantic import BaseModel, Field

from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from younger.commons.io import create_dir

from younger_apps_dl.commons.utils import get_device_descriptor, make_reproducible
from younger_apps_dl.commons.logging import logger, equip_logger
from younger_apps_dl.commons.checkpoint import load_checkpoint, save_checkpoint, Checkpoint

from younger_apps_dl.engines import BaseEngine, register_engine


class StandardTrainerOptions(BaseModel):
    # Checkpoint Options
    checkpoint_savepath: str = Field(..., description="Directory path to save checkpoint.")
    checkpoint_basename: str = Field('checkpoint', description="Base name of the checkpoint for save/load.")
    checkpoint_keepdisk: int = Field(5, ge=1, description="Number of checkpoints to keep on disk.")

    ## Resume Options
    resume_loadpath: str  = Field('', description="Path to load checkpoint. If "", train from scratch.")
    reset_iteration: bool = Field(True, description="Whether to reset the iteration status (epoch, step) when loading a checkpoint.")
    reset_optimizer: bool = Field(True, description="Whether to reset the optimizer when loading a checkpoint.")
    reset_scheduler: bool = Field(True, description="Whether to reset the scheduler when loading a checkpoint.")

    # Iteration Options
    seed: int = Field(3407, ge=0, description="Random seed for reproducibility.")
    shuffle: bool = Field(True, description="Shuffle the training data each epoch.")
    life_cycle: int = Field(100, ge=1, description="Lefe cycle of the training process (in epochs).")

    report_period: int = Field(100, ge=1, description="Period (in steps) to report the training status.")
    update_period: int = Field(1, ge=1, description="Period (in steps) to update the model parameters.")
    saving_period: int = Field(1000, ge=1, description="Period (in steps) to save the model parameters.")

    train_batch_size: int = Field(32, ge=1, description="Batch size for training.")
    valid_batch_size: int = Field(32, ge=1, description="Batch size for validation.")

    # Distribution Options
    distributed: bool = Field(False, description="Whether to use distributed training. If False, the options about distributed training will take no effect.")
    master_addr: str  = Field('localhost', description="Master address for distributed training.")
    master_port: str  = Field('16161', description="Master port for distributed training.")
    master_rank: int  = Field(0, ge=0, description="Master rank for distributed training. It should be < world_size and >= 0.")
    node_number: int  = Field(2, gt=1, description="Number of devices participating in distributed training. It should be > 1.")


@register_engine('trainer', 'standard')
class StandardTrainer(BaseEngine[StandardTrainerOptions]):
    _options_ = StandardTrainerOptions

    def __init__(
        self,
        configuration: dict,
        model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        train_fn: Callable[[torch.nn.Module, Any], tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]],
        valid_fn: Callable[[torch.nn.Module, Any], tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]],
        dataloader: Literal['pth', 'pyg'] = 'pth',
        logging_filepath: pathlib.Path | None = None,
    ):
        """
        _summary_

        :param configuration: _description_
        :type configuration: dict
        :param model: _description_
        :type model: torch.nn.Module
        :param optimizer: _description_
        :type optimizer: torch.optim.Optimizer
        :param scheduler: _description_
        :type scheduler: torch.optim.lr_scheduler.LRScheduler
        :param train_dataset: _description_
        :type train_dataset: torch.utils.data.Dataset
        :param valid_dataset: _description_
        :type valid_dataset: torch.utils.data.Dataset
        :param train_fn: A callable that defines how to process a `minibatch` during training, feeding it to the model and returning the computed metrics (names, values, formats). The 0th metric must be the loss to be optimized.
        :type train_fn: Callable[[Any], tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]]
        :param valid_fn: A callable that defines how to process a `sequence` of `minibatch` during validation, feeding it to the model and returning the computed metrics (names, values, formats).
        :type valid_fn: Callable[[Any], tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]]
        :param dataloader: _description_, defaults to 'pth'
        :type dataloader: Literal[&#39;pth&#39;, &#39;pyg&#39;], optional
        """

        super().__init__(configuration)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_fn = train_fn
        self.valid_fn = valid_fn
        self.dataloader = dataloader
        self.logging_filepath = logging_filepath
        self.start_from_epoch = 0
        self.start_from_step = 0
        self.start_from_itr = 0

    def log(self, epoch: int, step: int, itr: int, metric_names: list[str], metric_values: list[torch.Tensor], metric_formats: list[Callable[[float], str]]) -> None:
        with torch.no_grad():
            logs = list()
            for metric_name, metric_value, metric_format in zip(metric_names, metric_values, metric_formats):
                metric_value = metric_value.detach()
                distributed.all_reduce(metric_value, op = distributed.ReduceOp.SUM)
                logs.append(f'[{metric_name}]={metric_format(float(metric_value / self.options.node_number))}')
            logger.info(f'   [Epoch/Step/Itr]@[{epoch}/{step}/{itr}] - {' '.join(logs)}')

    def run(self) -> None:
        equip_logger(self.logging_filepath)
        if len(self.options.resume_filepath) == 0:
            logger.info(f'-> Train from scratch.')
        else:
            checkpoint = load_checkpoint(pathlib.Path(self.options.resume_loadpath))

            logger.info(f'-> Train from [Epoch/Step/Itr]@[{checkpoint.epoch}/{checkpoint.step}/{checkpoint.itr}].')

            if self.options.reset_iteration:
                logger.info(f'   Reset Epoch & Step & Itr.')
            else:
                self.start_from_epoch: int = checkpoint.epoch
                self.start_from_step: int = checkpoint.step
                self.start_from_itr: int = checkpoint.itr

            if self.options.reset_optimizer:
                logger.info(f'   Reset Optimizer.')
            else:
                self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)

            if self.options.reset_scheduler:
                logger.info(f'   Reset Scheduler.')
            else:
                self.scheduler.load_state_dict(checkpoint.scheduler_state_dict)

            logger.info(f'    v Loading Parameters ...')
            self.model.load_state_dict(checkpoint.model_state_dict)
            logger.info(f'    ^ Loaded.')

        create_dir(self.options.checkpoint_savepath)
        logger.info(f'-> Checkpoints will be saved into: \'{self.options.checkpoint_savepath}\'')

        logger.info(f'-> Training Start ...')
        logger.info(f'   Train Life Cycle: Total {self.options.life_cycle} Epochs!')
        logger.info(f'   Report every {self.options.report_period} Iteration (Itr);')
        logger.info(f'   Update every {self.options.update_period} Iteration (Itr);')
        logger.info(f'   Saving every {self.options.saving_period} Iteration (Itr);')

        tic = time.time()
        if self.options.distributed:
            master_rank = self.options.master_rank
            master_addr = self.options.master_addr
            master_port = self.options.master_port
            node_number = self.options.node_number

            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port

            assert torch.cuda.device_count() >= node_number, f'Insufficient GPU: {torch.cuda.device_count()}'
            assert master_rank < node_number, f'Wrong Master Rank: {master_rank}'

            logger.info(f'-> Distributed training - Total {node_number} GPU used.')

            torch.multiprocessing.spawn(self.pile_train, args=(), nprocs=node_number, join=True)
        else:
            self.solo_train()
        toc = time.time()
        logger.info(f'-> All Done! Overall Time Cost = {toc-tic:.2f}s')

    def pile_train(self, rank: int) -> None:
        equip_logger(self.logging_filepath)

        make_reproducible(self.options.seed)
        torch.autograd.set_detect_anomaly(True)

        device_descriptor = get_device_descriptor('GPU', rank)
        self.model.to(device=device_descriptor)
        logger.info(f'-> Process {rank} Use Device \'{device_descriptor}\'')

        master_flag = rank == self.options.master_rank
        if master_flag:
            logger.disabled = False
        else:
            logger.disabled = True

        distributed.init_process_group('nccl', rank=rank, world_size=self.options.node_number)
        model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=False)

        train_sampler = DistributedSampler(self.train_dataset, rank=rank, num_replicas=self.options.node_number, seed=self.options.seed, shuffle=self.options.shuffle, drop_last=True)

        if self.dataloader == 'pth':
            from torch.utils.data import DataLoader
        if self.dataloader == 'pyg':
            from torch_geometric.loader import DataLoader

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.options.train_batch_size, sampler=train_sampler)
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.options.valid_batch_size, shuffle=False)

        model.train()
        self.optimizer.zero_grad()
        itr = self.start_from_itr
        # Epoch 1 .. self.options.life_cycle
        for epoch in range(1, self.options.life_cycle + 1):
            if epoch < self.start_from_epoch:
                logger.info(f'-> Skip Epoch {epoch} Before Epoch {self.start_from_epoch}.')
                continue

            train_sampler.set_epoch(self.start_from_epoch)

            tic = time.time()
            # Step 1 .. len(train_dataloader)
            for step, minibatch in enumerate(train_dataloader, start=1):
                if epoch == self.start_from_epoch and step <= self.start_from_step:
                    logger.info(f'-> Skip Mini-Batch {step} Before Step {self.start_from_step}.')
                    continue

                itr = itr + 1

                train_metric_names, train_metric_values, train_metric_formats = self.train_fn(model, minibatch)
                assert len(train_metric_names) == len(train_metric_values) == len(train_metric_formats)

                # Update Model Parameters
                if itr % self.options.update_period == 0:
                    retain_graph = False
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    retain_graph = True
                train_metric_values[0].backward(retain_graph=retain_graph)

                # Report Metrics
                if itr % self.options.report_period == 0:
                    self.log(epoch, step, itr, train_metric_names, train_metric_values, train_metric_formats)

                # Validate and Save Model
                distributed.barrier()
                if itr % self.options.saving_period == 0 and master_flag:
                    logger.info(f'-> Validating ...')
                    model.eval()
                    stic = time.time()
                    valid_metric_names, valid_metric_values, valid_metric_formats = self.valid_fn(model, valid_dataloader)
                    stoc = time.time()
                    self.log(epoch, step, itr, valid_metric_names, valid_metric_values, valid_metric_formats)
                    model.train()
                    logger.info(f'   Time Cost: {stoc-stic:.2f}s')

                    logger.info(f'-> Saving ...')
                    stic = time.time()
                    checkpoint = Checkpoint(epoch, step, itr, model.module.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict(), dict(zip(valid_metric_names, valid_metric_values)))
                    save_checkpoint(checkpoint, self.options.checkpoint_savepath, self.options.checkpoint_basename, self.options.checkpoint_keepdisk)
                    stoc = time.time()
                    logger.info(f'   Time Cost: {stoc-stic:.2f}s')
                distributed.barrier()

            toc = time.time()
            logger.info(f'-> Epoch@{epoch} Finished. Time Cost = {toc-tic:.2f}s')

        distributed.destroy_process_group()

    def solo_train(self) -> None:
        equip_logger(self.logging_filepath)

        make_reproducible(self.options.seed)
        torch.autograd.set_detect_anomaly(True)

        device_descriptor = get_device_descriptor('GPU', 0)
        self.model.to(device=device_descriptor)

        train_sampler = RandomSampler(self.train_dataset) if self.options.shuffle else None
        if self.dataloader == 'pth':
            from torch.utils.data import DataLoader
        if self.dataloader == 'pyg':
            from torch_geometric.loader import DataLoader

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.options.train_batch_size, sampler=train_sampler)
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.options.valid_batch_size, shuffle=False)

        self.model.train()
        self.optimizer.zero_grad()
        itr = self.itr
        # Epoch 1 .. self.options.life_cycle
        for epoch in range(1, self.options.life_cycle + 1):
            if epoch < self.start_from_epoch:
                logger.info(f'-> Skip Epoch {epoch} Before Epoch {self.start_from_epoch}.')
                continue

            tic = time.time()
            # Step 1 .. len(train_dataloader)
            for step, minibatch in enumerate(train_dataloader, start=1):
                if epoch == self.start_from_epoch and step <= self.step:
                    logger.info(f'-> Skip Mini-Batch {step} Before Step {self.step}.')
                    continue

                itr = itr + 1

                train_metric_names, train_metric_values, train_metric_formats = self.train_fn(self.model, minibatch)
                assert len(train_metric_names) == len(train_metric_values) == len(train_metric_formats)

                # Update Model Parameters
                if itr % self.options.update_period == 0:
                    retain_graph = False
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    retain_graph = True
                train_metric_values[0].backward(retain_graph=retain_graph)

                # Report Metrics
                if itr % self.options.report_period == 0:
                    self.log(epoch, step, itr, train_metric_names, train_metric_values, train_metric_formats)

                # Validate and Save Model
                if itr % self.options.valid_period == 0:
                    logger.info(f'-> Validating ...')
                    self.model.eval()
                    stic = time.time()
                    valid_metric_names, valid_metric_values, valid_metric_formats = self.valid_fn(self.model, valid_dataloader)
                    stoc = time.time()
                    self.log(epoch, step, itr, valid_metric_names, valid_metric_values, valid_metric_formats)
                    self.model.train()
                    logger.info(f'   Time Cost: {stoc-stic:.2f}s')

                    logger.info(f'-> Saving checkpoint ...')
                    stic = time.time()
                    checkpoint = Checkpoint(epoch, step, itr, self.model.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict(), dict(zip(valid_metric_names, valid_metric_values)))
                    save_checkpoint(checkpoint, self.options.checkpoint_savepath, self.options.checkpoint_basename, self.options.checkpoint_keepdisk)
                    stoc = time.time()
                    logger.info(f'   Time Cost: {stoc-stic:.2f}s')

            toc = time.time()
            logger.info(f'-> Epoch@{epoch} Finished. Time Cost = {toc-tic:.2f}s')
