#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-16 22:58:32
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-15 15:17:56
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

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader

from younger.commons.io import create_dir, load_toml

from younger_apps_dl.tasks import standard_task_builders, StandardTask
from younger_apps_dl.commons.utils import get_model_parameters_number, get_device_descriptor, get_logging_metrics_str, fix_random_procedure, set_deterministic


def exact_eval(
    task: StandardTask,
    dataloader: DataLoader, 
    split: Literal['Valid', 'Test'],
):
    task.logger.info(f'-> {split} Begin ...')
    all_outputs = list()
    all_goldens = list()
    tic = time.time()
    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as progress_bar:
            for index, minibatch in enumerate(dataloader, start=1):
                eval_result = task.eval(minibatch)
                if eval_result is None:
                    pass
                else:
                    outputs, goldens = eval_result
                    all_outputs.append(outputs)
                    all_goldens.append(goldens)
                progress_bar.update(1)
    toc = time.time()

    logs = task.eval_calculate_logs(all_outputs, all_goldens)
    if logs is None:
        task.logger.info(f'-> {split} Finished. No User Defined Output')
    else:
        metrics = dict()
        for log_key, (log_value, log_format) in logs.items():
            metrics[log_key] = log_format(float(log_value))
        task.logger.info(f'-> {split} Finished. Overall Result - {get_logging_metrics_str(metrics)} (Time Cost = {toc-tic:.2f}s)')


def exact_train(
    rank: int,

    is_distribution: bool, master_rank: int, world_size: int, seed: int, make_deterministic: bool,

    task: StandardTask,

    checkpoint_dirpath: pathlib.Path, checkpoint_name: str, keep_number: int,

    train_batch_size: int, valid_batch_size: int, shuffle: bool,

    checkpoint_filepath: pathlib.Path, reset_optimizer: bool, reset_period: bool,

    life_cycle:int, report_period: int, update_period: int, train_period: int, valid_period: int,

    device: Literal['CPU', 'GPU'],
):
    device_descriptor = get_device_descriptor(device, rank)
    fix_random_procedure(seed)
    set_deterministic(make_deterministic)

    task.clean()
    task.to(device_descriptor)
    task.logger.info(f'-> Device \'{device_descriptor}\' Used')

    is_master = rank == master_rank
    torch.autograd.set_detect_anomaly(True)

    if is_master:
        task.logger.disabled = False
    else:
        task.logger.disabled = True

    task.logger.info(f'   Distribution: {is_distribution};{f" (Total {world_size} GPU)" if is_distribution else ""}')

    create_dir(checkpoint_dirpath)
    task.logger.info(f'-> Checkpoints will be saved into: \'{checkpoint_dirpath}\'')

    # Build Model
    task.logger.info(f'-> Preparing Model ...')

    # Print Model
    task.logger.info(f'-> Model Specs:')
    parameters_number = get_model_parameters_number(task.model)
    parameters_number_str = str()
    for name, number in parameters_number.items():
        parameters_number_str += f'{name}: {number} Elements ;\n'
    parameters_number_str += f'Total: {sum(parameters_number.values())} Elements .\n'
    task.logger.info(
        f'\n======= v Model Architecture v ======='
        f'\n{task.model}'
        f'\n'
        f'\n====== v Number of Parameters v ======'
        f'\n{parameters_number_str}'
    )

    # Model
    if is_distribution:
        distributed.init_process_group('nccl', rank=rank, world_size=world_size)
        task.model = torch.nn.parallel.DistributedDataParallel(task.model, device_ids=[rank], find_unused_parameters=False)

    # Datasets

    # Training Dataset
    if is_distribution:
        train_sampler = DistributedSampler(task.train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=True)
    else:
        train_sampler = RandomSampler(task.train_dataset) if shuffle else None
    train_dataloader = DataLoader(task.train_dataset, batch_size=train_batch_size, sampler=train_sampler)

    # Validation Dataset
    if task.valid_dataset:
        valid_dataloader = DataLoader(task.valid_dataset, batch_size=valid_batch_size, shuffle=False)
    else:
        valid_dataloader = None

    # Init Train Status
    if checkpoint_filepath:
        checkpoint = load_checkpoint(pathlib.Path(checkpoint_filepath), checkpoint_name)
    else:
        checkpoint = None

    if checkpoint is None:
        task.logger.info(f'-> Train from scratch.')
        start_position = 0
    else:
        task.logger.info(f'-> Train from checkpoint [\'{checkpoint_filepath}\'] [Epoch/Step]@[{checkpoint["Epoch"]}/{checkpoint["Step"]}].')

        if reset_optimizer:
            task.logger.info(f'   Reset Optimizer.')
        else:
            task.optimizer.load_state_dict(checkpoint['optimizer_state'])

        task.logger.info(f'    v Loading Parameters ...')
        task.model.load_state_dict(checkpoint['model_state'])
        task.logger.info(f'    ^ Loaded.')

        if reset_period:
            task.logger.info(f'   Reset Epoch & Step.')
            start_position = 0
        else:
            start_position = checkpoint['Step']

    task.logger.info(f'-> Training Start ...')
    task.logger.info(f'   Train Life Cycle: Total {life_cycle} Epochs!')
    task.logger.info(f'   Update every {update_period} Step;')
    task.logger.info(f'   Report every {report_period} Step;')
    task.logger.info(f'   Validate every {valid_period} Step;')
    task.logger.info(f'   Save checkpoint every {train_period} Step.')

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


def train(
    configuration_filepath: pathlib.Path,

    checkpoint_dirpath: pathlib.Path, checkpoint_name: str = 'checkpoint', keep_number: int = 50,

    train_batch_size: int = 32, valid_batch_size: int = 32, shuffle: bool = True,

    checkpoint_filepath: str | None = None, reset_optimizer: bool = True, reset_period: bool = True,

    life_cycle: int = 100, report_period: int = 100, update_period: int = 1, train_period: int = 1000, valid_period: int = 1000,

    device: Literal['CPU', 'GPU'] = 'GPU',

    world_size: int = 1, master_addr: str = 'localhost', master_port: str = '16161', master_rank: int = 0,

    seed: int = 1234, make_deterministic: bool = False,
):
    assert task_name in standard_task_builders, f'Standard Task ({task_name}) is not Defined'

    configuration = load_toml(configuration_filepath)
    task: StandardTask = standard_task_builders[task_name](configuration)

    task.logger.info(f'-> Task: \'{task_name}\' | Configuration Loaded From {configuration_filepath}')

    task.logger.info(f'-> Preparing Datasets ...')

    task.logger.info(f'   Dataset Split Sizes:')
    task.logger.info(f'    * Train - {len(task.train_dataset)}')
    if task.valid_dataset:
        task.logger.info(f'    * Valid - {len(task.valid_dataset)}')
    else:
        task.logger.info(f'    * Valid - Not Provided')

    assert device in {'CPU', 'GPU'}
    if device == 'CPU':
        is_distribution = False
    if device == 'GPU':
        assert torch.cuda.device_count() >= world_size, f'Insufficient GPU: {torch.cuda.device_count()}'
        assert master_rank < world_size, f'Wrong Master Rank: {master_rank}'
        is_distribution = False if world_size == 1 else True

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    torch.multiprocessing.spawn(
        exact_train,
        args=(
            is_distribution, master_rank, world_size, seed, make_deterministic,

            task,

            checkpoint_dirpath, checkpoint_name, keep_number,

            train_batch_size, valid_batch_size, shuffle,

            checkpoint_filepath, reset_optimizer, reset_period,

            life_cycle, report_period, update_period, train_period, valid_period,

            device,
        ),
        nprocs=world_size,
        join=True
    )
