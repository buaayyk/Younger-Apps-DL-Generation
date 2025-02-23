#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-16 22:58:32
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-02-23 22:55:42
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import tqdm
import time
import torch
import pathlib

from torch import distributed
from typing import Literal
from pydantic import BaseModel, Field

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, RandomSampler
from torch_geometric.loader import DataLoader

from younger.commons.io import create_dir, load_toml

from younger_apps_dl.commons.logging import logger, equip_logger
from younger_apps_dl.commons.utils import load_checkpoint, save_checkpoint, get_model_parameters_number, get_device_descriptor, get_logging_metrics_str, make_reproducible

from younger_apps_dl.engines import BaseEngine


class DistributedTrainerOptions(BaseModel):
    # Main Options
    logging_filepath: str = Field('./default_distributed_trainer.log', description="Logging file path where logs will be saved, default to None, which may save to a default path that is determined by the Younger.")

    # Distribution Options
    master_addr: str = Field('localhost', description="Master address for distributed training.")
    master_port: str = Field('16161', description="Master port for distributed training.")
    master_rank: int = Field(0, ge=0, description="Master rank for distributed training. It should be < world_size and >= 0.")
    world_size: int = Field(2, gt=1, description="Number of devices participating in distributed training. It should be > 1.")

    # Checkpoint Options
    checkpoint_basename: str = Field('checkpoint', description="Base name of the checkpoint for save/load.")

    ## Save
    checkpoint_save_dirpath: str | None = Field(None, description="Directory path for saving checkpoints.")
    checkpoint_save_number: int = Field(5, ge=1, description="Number of checkpoints to save on disk.")
    checkpoint_save_metric: str | None = Field(None, description="Metric name for sorting the checkpoints. If None, it will use the latest N checkpoints.")

    ## Load
    checkpoint_load_dirpath: str | None = Field(None, description="Directory path for loading checkpoints.")
    checkpoint_load_number: int = Field(5, ge=1, description="Number of checkpoints to load from disk.")
    checkpoint_load_metric: str | None = Field(None, description="Metric name for sorting the checkpoints. If None, it will use the latest N checkpoints.")

    ## Reset
    reset_iteration: bool = Field(True, description="Whether to reset the iteration status (epoch, step) when loading a checkpoint.")
    reset_optimizer: bool = Field(True, description="Whether to reset the optimizer when loading a checkpoint.")
    reset_scheduler: bool = Field(True, description="Whether to reset the scheduler when loading a checkpoint.")

    # Iteration Options
    seed: int = Field(3407, ge=0, description="Random seed for reproducibility.")
    shuffle: bool = Field(True, description="Shuffle the training data each epoch.")
    life_cycle: int = Field(100, ge=1, description="Lefe cycle of the training process (in epochs).")

    report_period: int = Field(100, ge=1, description="Period (in steps) to report the training status.")
    update_period: int = Field(1, ge=1, description="Period (in steps) to update the model parameters.")

    train_period: int = Field(1000, ge=1, description="Period (in steps) to save the model parameters.")
    valid_period: int = Field(1000, ge=1, description="Period (in steps) to validate the model.")
    train_batch_size: int = Field(32, ge=1, description="Batch size for training.")
    valid_batch_size: int = Field(32, ge=1, description="Batch size for validation.")


class DistributedTrainer(BaseEngine[DistributedTrainerOptions]):
    _options_ = DistributedTrainerOptions

    def __init__(
        self,
        configuration: dict,
        model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_dataset: torch.utils.data.Dataset, valid_dataset: torch.utils.data.Dataset,
    ):
        super().__init__(configuration)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.position = 0

    def run(self):
        master_rank = self.options.master_rank
        master_addr = self.options.master_addr
        master_port = self.options.master_port

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

        world_size = self.options['world_size']
        assert torch.cuda.device_count() >= world_size, f'Insufficient GPU: {torch.cuda.device_count()}'
        assert master_rank < world_size, f'Wrong Master Rank: {master_rank}'

        logger.info(f'-> Distributed training - Total {world_size} GPU used.')

        if self.options.checkpoint_load_dirpath is None:
            logger.info(f'-> Train from scratch.')
        else:
            checkpoint = load_checkpoint(
                pathlib.Path(self.options.checkpoint_load_dirpath),
                self.options.checkpoint_basename,
                self.options.checkpoint_load_number,
                self.options.checkpoint_load_metric
            )

            logger.info(f'-> Train from [Epoch/Step]@[{checkpoint["Epoch"]}/{checkpoint["Step"]}].')

            if self.options.reset_iteration:
                logger.info(f'   Reset Epoch & Step.')
            else:
                self.position = checkpoint['Step']

            if self.options.reset_optimizer:
                logger.info(f'   Reset Optimizer.')
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            if self.options.reset_scheduler:
                logger.info(f'   Reset Scheduler.')
            else:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])

            logger.info(f'    v Loading Parameters ...')
            self.model.load_state_dict(checkpoint['model_state'])
            logger.info(f'    ^ Loaded.')

        create_dir(self.options.checkpoint_save_dirpath)
        logger.info(f'-> Checkpoints will be saved into: \'{self.options.checkpoint_save_dirpath}\'')

        logger.info(f'-> Training Start ...')
        logger.info(f'   Train Life Cycle: Total {self.options.life_cycle} Epochs!')
        logger.info(f'   Report every {self.options.report_period} Step;')
        logger.info(f'   Update every {self.options.update_period} Step;')
        logger.info(f'   Save CPT every {self.options.train_period} Step;')
        logger.info(f'   Validate every {self.options.valid_period} Step.')

        torch.multiprocessing.spawn(self.train, args=(), nprocs=world_size, join=True)

    def train(self, rank: int):
        equip_logger(self.options.logging_filepath)

        make_reproducible(self.options.seed)
        torch.autograd.set_detect_anomaly(True)

        device_descriptor = get_device_descriptor('GPU', rank)
        self.model.to(device=device_descriptor)
        logger.info(f'-> Process {rank} Use Device \'{device_descriptor}\'')

        if rank == self.options.master_rank:
            logger.disabled = False
        else:
            logger.disabled = True

        checkpoint_save_number = self.options.checkpoint_save_number
        checkpoint_save_metric = self.options.checkpoint_save_metric

        distributed.init_process_group('nccl', rank=rank, world_size=self.options.world_size)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

        train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=True)
        train_dataloader = DataLoader(self.train_dataset, batch_size=train_batch_size, sampler=train_sampler)

        # Validation Dataset
        if task.valid_dataset:
            valid_dataloader = DataLoader(task.valid_dataset, batch_size=valid_batch_size, shuffle=False)
        else:
            valid_dataloader = None

        task.model.train()
        task.optimizer.zero_grad()
        epoch = 0
        step = start_position
        while epoch < life_cycle:
            if is_distribution:
                train_sampler.set_epoch(epoch)

            tic = time.time()
            for minibatch in train_dataloader:
                (loss, logs) = task.train(minibatch)

                # Report Metrics
                if step % report_period == 0:
                    metrics = dict()
                    for log_key, (log_value, log_format) in logs.items():
                        if is_distribution:
                            distributed.all_reduce(log_value, op = distributed.ReduceOp.SUM)
                            log_value = log_value / world_size
                        metrics[log_key] = log_format(float(log_value))
                    task.logger.info(f'   [Epoch/Step]@[{epoch}/{step}] - {get_logging_metrics_str(metrics)}')

                # Update Model Parameters
                if step % update_period == 0:
                    retain_graph = False
                    task.optimizer.step()
                    task.optimizer.zero_grad()
                else:
                    retain_graph = True
                loss.backward(retain_graph=retain_graph)

                # Save Model Parameters
                if step % train_period == 0 and is_master:
                    task.logger.info('-> Saving checkpoint ...')
                    tic = time.time()
                    task.save(mode='Train', checkpoint_path=checkpoint_dirpath, checkpoint_name=checkpoint_name, keep_number=keep_number)
                    toc = time.time()
                    task.logger.info(f'-> Checkpoint is saved to \'{checkpoint_dirpath}\' at [Epoch/Step][{epoch}/{step}] (Time Cost: {toc-tic:.2f}s)')        

                # Do Validation
                if step % valid_period == 0:
                    if is_distribution:
                        distributed.barrier()
                    if is_master:
                        if valid_dataloader:
                            task.model.eval()
                            exact_eval(
                                task,
                                valid_dataloader, 
                                'Valid',
                            )
                            task.save(mode='Valid', checkpoint_path=checkpoint_dirpath, checkpoint_name=checkpoint_name, keep_number=keep_number)
                            task.model.train()
                    if is_distribution:
                        distributed.barrier()

                if task.done:
                    task.logger.info(f'-> [Inner Epoch] The termination condition is triggered, stopping the current training process.')
                    break

                step += 1
                task.update_status(stage='Step', step=step, epoch=epoch, loss=loss)

            if task.done:
                task.logger.info(f'-> [Inter Epoch] The termination condition is triggered, stopping the current training process.')
                break

            toc = time.time()
            task.logger.info(f'-> Epoch@{epoch} Finished. Time Cost = {toc-tic:.2f}s')

            epoch += 1
            task.update_status(stage='Epoch', step=step, epoch=epoch, loss=loss)

        if is_distribution:
            distributed.destroy_process_group()
