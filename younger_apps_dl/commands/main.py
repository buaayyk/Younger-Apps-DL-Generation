#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-11-27 09:53:09
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-17 09:36:44
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib
import tabulate

from typing import Literal

from younger.commons.io import load_toml, save_toml

from younger_apps_dl.commands import equip_logger


@click.group(name='younger-apps-dl')
def main():
    pass


@main.command(name='glance')
@click.option('--some-type', type=click.Choice(['components', 'engines', 'tasks'], case_sensitive=True), help='Indicates the type of task will be used.')
def glance(some_type: Literal['components', 'engines', 'tasks']):
    """
    Displays all possible `kind` and `name` candidates under a specific `type`.

    :param some_type: _description_
    :type some_type: Literal[&#39;components&#39;, &#39;engines&#39;, &#39;tasks&#39;]
    """
    table_name = some_type.capitalize()
    table_data = list()
    registry = dict()
    if some_type == 'components':
        from younger_apps_dl.components import COMPONENT_REGISTRY
        registry = COMPONENT_REGISTRY

    if some_type == 'engines':
        from younger_apps_dl.engines import ENGINE_REGISTRY
        registry = ENGINE_REGISTRY

    if some_type == 'tasks':
        from younger_apps_dl.tasks import TASK_REGISTRY
        registry = TASK_REGISTRY

    for kind, name2cls in registry.items():
        for name, cls in name2cls.items():
            table_data.append([kind, name, cls.__name__])

    if len(table_data) == 0:
        print(f'{table_name}\'s Registry is Empty')
    else:
        print(f'{table_name}\'s Registry')
        print(tabulate.tabulate(table_data, headers=['Kind', 'Name', 'Class'], tablefmt='grid'))


@main.command(name='option')
@click.option('--some-type', type=click.Choice(['components', 'engines', 'tasks'], case_sensitive=True), help='Indicates one of the core modules of the project.')
@click.option('--some-kind', type=str, help='Indicates a specific category within a particular `type`.')
@click.option('--some-name', type=str, help='Indicates a specific class definition within a particular `kind` under a specific `type`.')
def option(some_type: Literal['components', 'engines', 'tasks'], some_kind: str, some_name: str):
    """
    Displays the configuration options corresponding to a specific combination of <`type`, `kind`, `name`>.

    :param some_type: _description_
    :type some_type: Literal[&#39;components&#39;, &#39;engines&#39;, &#39;tasks&#39;]
    :param some_kind: _description_
    :type some_kind: str
    :param some_name: _description_
    :type some_name: str
    """


@main.command(name='launch')
@click.option('--task-kind',        required=True,  type=str, help='Indicates the type of task.')
@click.option('--task-name',        required=True,  type=str, help='Indicates the name of task.')
@click.option('--task-step',        required=True,  type=click.Choice(['train', 'evaluate', 'predict', 'preprocess', 'postprocess'], case_sensitive=True), help='Indicates the step of task.')
@click.option('--options-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the options file; if not provided, default options will be used.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def train(task_kind, task_name, task_step, options_filepath, logging_filepath):
    """
    _summary_

    :param task_kind: _description_
    :type task_kind: _type_
    :param task_name: _description_
    :type task_name: _type_
    :param task_step: _description_
    :type task_step: _type_
    :param options_filepath: _description_
    :type options_filepath: _type_
    :param logging_filepath: _description_
    :type logging_filepath: _type_
    :raises exception: _description_
    """

    equip_logger(logging_filepath)

    from younger_apps_dl.tasks import TASK_REGISTRY
    try:
        Task = TASK_REGISTRY[task_kind][task_name]
    except Exception as exception:
        click.echo(f'No <{task_kind}, {task_name}> Task in Task Registry.')
        raise exception

    task = Task(load_toml(options_filepath))
    if task_step == 'train':
        task.train()

    if task_step == 'evaluate':
        task.evaluate()

    if task_step == 'predict':
        task.predict()

    if task_step == 'preprocess':
        task.preprocess()

    if task_step == 'postprocess':
        task.postprocess()


if __name__ == '__main__':
    main()
