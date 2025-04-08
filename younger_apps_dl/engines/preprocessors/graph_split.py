#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-07 22:21:54
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import numpy
import pathlib
import networkx
import collections

from typing import Any
from pydantic import BaseModel, Field

from younger.commons.io import save_json, save_pickle, create_dir, tar_archive

from younger_apps_dl.commons.logging import logger, equip_logger

from younger_apps_dl.engines import BaseEngine, register_engine


class GraphSplitOptions(BaseModel):
    load_dirpath: str = Field(..., description='Directory path to load LogicX\'s.')
    save_dirpath: str = Field(..., description='Directory path to save LogicX\'s.')

    split_scale: list[int] = Field(..., description='List of node counts to include in subgraph splits expanded from central nodes. Each value specifies a different subgraph split scale to generate.')
    split_count: int = Field(..., description='Number of subgraph splits to generate per central node.')
    split_tries: int = Field(..., description='Maximum number of attempts to generate `split_count` valid subgraphs (e.g., avoiding duplicates or undersized splits).')

    training_dataset_size: int = Field(..., 'Number of subgraphs splits to include in the training set.')
    validation_dataset_size: int = Field(..., 'Number of subgraphs splits to include in the validation set.')
    test_dataset_size: int = Field(..., 'Number of subgraph splits to include in the test set.')

    min_graph_size: int | None = Field(None, ge=0, description='Minimum number of nodes a full graph must have to be considered for graph split. '
                                                               'Graphs smaller than this value will be excluded. '
                                                               'Set to `None` to disable this filter.')
    max_graph_size: int | None = Field(None, ge=0, description='Maximum number of nodes a full graph must have to be considered for graph split. '
                                                               'Graphs larger than this value will be excluded. '
                                                               'Set to `None` to disable this filter.')

    seed: int = Field(16861, ge=0, description='Random seed for deterministic behavior during subgraph split sampling.')


@register_engine('preprocessor', 'graph_split')
class GraphSplit(BaseEngine[GraphSplitOptions]):
    _options_ = GraphSplitOptions

    def __init__(self,
        configuration: dict,
        logging_filepath: pathlib.Path | None = None,
    ):
        super().__init__(configuration)
        self.logging_filepath = logging_filepath

    def run(self) -> None:
        equip_logger(self.logging_filepath)

        numpy.random.seed(self.options.seed)

        logger.info(f'')

        load_dirpath = pathlib.Path(self.options.load_dirpath)
        logicx_filepaths = sorted([logicx_filepath for logicx_filepath in load_dirpath.iterdir()])

        logger.info(f'Scan Load Directory & Generate Node UUID List ...')
        instances: list[Instance] = list() # [ins1, ins2, ...]
        all_uuid_positions: dict[str, dict[int, set[str]]] = dict() # {sop_identifier: {instance_index: set[node_index]}}
        # All Tasks of Each Instance
        tasks_of_instances: list[list[int]] = list()
        with tqdm.tqdm(total=len(paths)) as progress_bar:
            instance_index = 0
            for path in paths:
                progress_bar.update(1)
                instance = Instance()
                instance.load(path)
                tasks_of_instance = set([YoungerDatasetTask.T2I[tag] for tag in instance.labels['tags'] if tag in YoungerDatasetTask.T2I])
                graph_node_num, graph_edge_num = instance.network.graph.number_of_nodes(), instance.network.graph.number_of_edges()
                node_size_lbound = graph_node_num if global_node_size_lbound is None else global_node_size_lbound
                node_size_ubound = graph_node_num if global_node_size_ubound is None else global_node_size_ubound

                edge_size_lbound = graph_edge_num if global_edge_size_lbound is None else global_edge_size_lbound
                edge_size_ubound = graph_edge_num if global_edge_size_ubound is None else global_edge_size_ubound
                if graph_node_num < node_size_lbound or node_size_ubound < graph_node_num or graph_edge_num < edge_size_lbound or edge_size_ubound < graph_edge_num:
                    continue

                instances.append(instance)
                tasks_of_instances.append(tasks_of_instance)

                for node_index in instance.network.graph.nodes():
                    node_features = instance.network.graph.nodes[node_index]['features']
                    node_origin = get_operator_origin(node_features['operator']['op_type'], domain=node_features['operator']['domain'])
                    if node_origin == 'onnx' or node_origin in allow_domains:
                        sop_identifier = Network.get_node_identifier_from_features(node_features, mode='type')
                        sop_positions = all_sop_positions.get(sop_identifier, dict())
                        sop_node_indices = sop_positions.get(instance_index, set())
                        sop_node_indices.add(node_index)
                        sop_positions[instance_index] = sop_node_indices
                        all_sop_positions[sop_identifier] = sop_positions
                instance_index += 1

        sop_dict: dict[str, int] = dict()
        for sop_name, sop_positions in all_sop_positions.items():
            sop_dict[sop_name] = sum([len(node_indices) for instance_index, node_indices in sop_positions.items()])
        logger.info(f'Total {len(sop_dict)} Standard ONNX Operators')

        logger.info(f'User Specified # of Splits (Each Size, Each Op) - Training/Validation/Test = {self.options.training_dataset_size} / {self.options.validation_dataset_size} / {self.options.test_dataset_size}')

        logger.info(f'Expected Split Number (Overall) - Train/Valid/Test = {len(sop_dict)*train_number*len(subgraph_sizes)} / {len(sop_dict)*valid_number*len(subgraph_sizes)} / {len(sop_dict)*test_number*len(subgraph_sizes)}')
        logger.info(f'Retrieving All Subgraphs ...')
        subgraphs: dict[int, dict[str, tuple[networkx.DiGraph, tuple[str, ]]]] = dict() # {subgraph_size: {wl_hash: (networkx.DiGraph, (node_index, ))}}
        subgraph_hashes: dict[int, dict[str, list[str]]] = dict() # {subgraph_size: {sop_name: list[wl_hash]}}
        # Select for each Subgraph Size:
        for subgraph_size in subgraph_sizes:
            # Select for each Standard ONNX Operator:
            logger.info(f' -> Now Retrieving All Subgraphs with Size {subgraph_size}...')
            subgraphs[subgraph_size] = dict()
            subgraph_hashes_at_size: dict[str, list[str]] = dict()
            with tqdm.tqdm(total=len(all_sop_positions)) as progress_bar:
                for sop_name, sop_positions in all_sop_positions.items():
                    subgraph_hashes_at_size[sop_name] = list()

                    current_generated = 0

                    instance_indices: set[int] = set(sop_positions.keys())
                    # Generate subgraph repeatedly
                    total_try = 0
                    while len(instance_indices) != 0 and current_generated < subgraph_number:
                        total_try += 1
                        if total_try >= retrieve_try:
                            break
                        instance_index: int = int(numpy.random.choice(list(instance_indices)))
                        node_indices = sop_positions[instance_index]
                        node_index: str = str(numpy.random.choice(list(node_indices)))
                        subgraph = retrieve_subgraph(instances[instance_index].network.graph, node_index, subgraph_size)
                        subgraph.graph['graph_hash'] = instances[instance_index].labels['hash']
                        if subgraph.number_of_nodes() < subgraph_size:
                            # all_sop_positions[sop_name][instance_index].remove(node_index)
                            # if len(all_sop_positions[sop_name][instance_index]) == 0:
                            #     all_sop_positions[sop_name].pop(instance_index)
                            #     instance_indices.remove(instance_index)
                            continue
                        else:
                            subgraph_hash = Network.hash(subgraph, node_attr='operator')
                            subgraph.graph['tasks'] = tasks_of_instances[instance_index]
                            if subgraph_hash in subgraphs[subgraph_size]:
                                continue
                            else:
                                subgraphs[subgraph_size][subgraph_hash] = (subgraph, (node_index,))
                                current_generated += 1
                                # Add To Split
                                subgraph_hashes_at_size[sop_name].append(subgraph_hash)
                    if current_generated < subgraph_number:
                        flag = f'No!'
                    else:
                        flag = f'Ye!'
                    progress_bar.set_description(f'Current: {sop_name} Enough: ({flag})')
                    progress_bar.update(1)
            subgraph_hashes[subgraph_size] = subgraph_hashes_at_size

        subgraph_train_split = [
            (subgraph_hash, subgraphs[subgraph_size][subgraph_hash][0], subgraphs[subgraph_size][subgraph_hash][1])
            for subgraph_size, subgraph_hashes_at_size in subgraph_hashes.items()
            for sop_name, subgraph_hashes_list in subgraph_hashes_at_size.items()
            for index, subgraph_hash in enumerate(subgraph_hashes_list)
            if index < int(len(subgraph_hashes_list)*train_ratio)
        ]

        subgraph_valid_split = [
            (subgraph_hash, subgraphs[subgraph_size][subgraph_hash][0], subgraphs[subgraph_size][subgraph_hash][1])
            for subgraph_size, subgraph_hashes_at_size in subgraph_hashes.items()
            for sop_name, subgraph_hashes_list in subgraph_hashes_at_size.items()
            for index, subgraph_hash in enumerate(subgraph_hashes_list)
            if int(len(subgraph_hashes_list)*train_ratio) <= index and index < int(len(subgraph_hashes_list)*(train_ratio + valid_ratio))
        ]

        subgraph_test_split = [
            (subgraph_hash, subgraphs[subgraph_size][subgraph_hash][0], subgraphs[subgraph_size][subgraph_hash][1])
            for subgraph_size, subgraph_hashes_at_size in subgraph_hashes.items()
            for sop_name, subgraph_hashes_list in subgraph_hashes_at_size.items()
            for index, subgraph_hash in enumerate(subgraph_hashes_list)
            if int(len(subgraph_hashes_list)*(train_ratio + valid_ratio)) <= index
        ]

        logger.info(f'Dict Size = {len(sop_dict)}')
        logger.info(f'Exact Split Number (Overall) - Train/Valid/Test = {len(subgraph_train_split)} / {len(subgraph_valid_split)} / {len(subgraph_test_split)}')

        meta = dict(
            all_operators = sop_dict,
            all_tasks = YoungerDatasetTask.I2T,
        )

        if len(subgraph_train_split):
            train_split_meta = dict(
                split = f'train',
                archive = f'train.tar.gz',
                version = f'{version}',
                size = len(subgraph_train_split),
                url = "",
            )
            train_split_meta.update(meta)
            save_split(train_split_meta, subgraph_train_split, save_dirpath)
        else:
            logger.info(f'No Instances in Train Split')

        if len(subgraph_valid_split):
            valid_split_meta = dict(
                split = f'valid',
                archive = f'valid.tar.gz',
                version = f'{version}',
                size = len(subgraph_valid_split),
                url = "",
            )
            valid_split_meta.update(meta)
            save_split(valid_split_meta, subgraph_valid_split, save_dirpath)
        else:
            logger.info(f'No Instances in Valid Split')

        if len(subgraph_valid_split):
            test_split_meta = dict(
                split = f'test',
                archive = f'test.tar.gz',
                version = f'{version}',
                size = len(subgraph_test_split),
                url = "",
            )
            test_split_meta.update(meta)
            save_split(test_split_meta, subgraph_test_split, save_dirpath)
        else:
            logger.info(f'No Instances in Test Split')

def save_item(parameter: tuple[pathlib.Path, tuple[str, tuple[networkx.DiGraph, set]]]):
    save_filepath, item = parameter
    save_pickle(item, save_filepath)


def save_split(meta: dict[str, Any], selected_subgraph_with_labels: list[tuple[str, tuple[networkx.DiGraph, set]]], save_dirpath: pathlib.Path):
    version_dirpath = save_dirpath.joinpath(meta['version'])
    split_dirpath = version_dirpath.joinpath(meta['split'])
    archive_filepath = split_dirpath.joinpath(meta['archive'])
    split = meta['split']

    item_dirpath = split_dirpath.joinpath('item')
    meta_filepath = split_dirpath.joinpath('meta.json')
    create_dir(item_dirpath)

    logger.info(f'Saving \'{split}\' Split META {meta_filepath.absolute()} ... ')
    save_json(meta, meta_filepath, indent=2)
    logger.info(f'Saved.')

    logger.info(f'Saving \'{split}\' Split {item_dirpath.absolute()} ... ')
    parameters = [
        (
            item_dirpath.joinpath(f'sample-{i}.pkl'),
            item,
        ) for i, item in enumerate(selected_subgraph_with_labels)
    ]
    with tqdm.tqdm(total=len(parameters), desc='Saving') as progress_bar:
        for index, parameter in enumerate((parameters), start=1):
            save_item(parameter)
            progress_bar.update(1)
    logger.info(f'Saved.')

    logger.info(f'Saving \'{split}\' Split Tar {archive_filepath.absolute()} ... ')
    tar_archive(
        [item_dirpath.joinpath(f'sample-{i}.pkl') for i, _ in enumerate(selected_subgraph_with_labels)],
        archive_filepath,
        compress=True
    )
    logger.info(f'Saved.')


def retrieve_subgraph(graph: networkx.DiGraph, node_index: str, subgraph_size: int) -> networkx.DiGraph:
    bfs_flags = set([node_index])
    bfs_queue = collections.deque([node_index])
    while len(bfs_queue) != 0 and len(bfs_flags) < subgraph_size:
        vertex = bfs_queue.popleft()
        neighbors = list()

        for neighbor in networkx.function.all_neighbors(graph, vertex):
            # node_features = graph.nodes[neighbor]['features']
            # node_origin = get_operator_origin(node_features['operator']['op_type'], domain=node_features['operator']['domain'])
            # if node_origin == 'onnx':
            #     neighbors.append(neighbor)
            neighbors.append(neighbor)
        if len(neighbors) == 0:
            continue

        numpy.random.shuffle(neighbors)
        limit = numpy.random.randint(1, len(neighbors) + 1)
        for neighbor in neighbors[:limit]:
            if len(bfs_flags) < subgraph_size and neighbor not in bfs_flags:
                bfs_flags.add(neighbor)
                bfs_queue.append(neighbor)

    subgraph = networkx.subgraph(graph, bfs_flags).copy()

    cleansed_subgraph = networkx.DiGraph()
    cleansed_subgraph.add_nodes_from(subgraph.nodes(data=False))
    cleansed_subgraph.add_edges_from(subgraph.edges(data=False))
    # cleansed_subgraph.add_nodes_from(subgraph.nodes(data=True))
    # cleansed_subgraph.add_edges_from(subgraph.edges(data=True))
    for node_index in cleansed_subgraph.nodes():
        cleansed_subgraph.nodes[node_index]['operator'] = subgraph.nodes[node_index]['features']['operator']

    return cleansed_subgraph
