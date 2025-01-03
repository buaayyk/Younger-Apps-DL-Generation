#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-04 12:16
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
import argparse


def run(arguments):
    pass


def deep_learning_run(arguments):
    pass

def deep_learning_test_run(arguments):
    task_name = arguments.task_name
    config_filepath = pathlib.Path(arguments.config_filepath)
    checkpoint_filepath = pathlib.Path(arguments.checkpoint_filepath)

    from younger.applications.deep_learning import test

    test(
        task_name,
        config_filepath,
        checkpoint_filepath,
        arguments.test_batch_size,
        arguments.device,
    )


def deep_learning_cli_run(arguments):
    task_name = arguments.task_name
    config_filepath = pathlib.Path(arguments.config_filepath)
    checkpoint_filepath = pathlib.Path(arguments.checkpoint_filepath)

    from younger.applications.deep_learning import cli

    cli(
        task_name,
        config_filepath,
        checkpoint_filepath,
        arguments.device,
    )


def deep_learning_api_run(arguments):
    task_name = arguments.task_name
    config_filepath = pathlib.Path(arguments.config_filepath)
    checkpoint_filepath = pathlib.Path(arguments.checkpoint_filepath)

    from younger.applications.deep_learning import api

    api(
        task_name,
        config_filepath,
        checkpoint_filepath,
        arguments.device,
    )


def set_applications_deep_learning_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    train_parser = subparser.add_parser('train')
    test_parser = subparser.add_parser('test')
    test_parser.add_argument('--task-name', type=str)
    test_parser.add_argument('--config-filepath', type=str)

    test_parser.add_argument('--checkpoint-filepath', type=str)
    test_parser.add_argument('--test-batch-size', type=int, default=32)

    test_parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU')
    test_parser.set_defaults(run=deep_learning_test_run)

    api_parser = subparser.add_parser('cli')
    api_parser.add_argument('--task-name', type=str)
    api_parser.add_argument('--config-filepath', type=str)
    api_parser.add_argument('--checkpoint-filepath', type=str)
    api_parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU')
    api_parser.set_defaults(run=deep_learning_cli_run)


    api_parser = subparser.add_parser('api')
    api_parser.add_argument('--task-name', type=str)
    api_parser.add_argument('--config-filepath', type=str)
    api_parser.add_argument('--checkpoint-filepath', type=str)
    api_parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU')
    api_parser.set_defaults(run=deep_learning_api_run)

    parser.set_defaults(run=deep_learning_run)


def set_applications_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers()

    deep_learning_parser = subparser.add_parser('deep_learning')

    set_applications_deep_learning_arguments(deep_learning_parser)

    parser.set_defaults(run=run)
