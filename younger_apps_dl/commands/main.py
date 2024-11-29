#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-11-27 09:53:09
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-11-27 14:15:23
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click

from younger.commons.logging import naive_log

from younger_apps_dl.commands.standard import standard


@click.group()
def main():
    naive_log(
        f'                                                                \n'
        f'                >   Welcome to use Younger-App-DL!   <          \n'
        f'----------------------------------------------------------------\n'
        f'Please use the following command to make the most of the system:\n'
        f'0. younger-app-dl --help                                               \n'
        f'1. younger-app-dl apps --help                                          \n'
        f'2. younger-app-dl tools --help                                         \n'
        f'3. younger-app-dl logics --help                                        \n'
        f'                                                                \n'
    )


main.add_command(standard)

if __name__ == '__main__':
    main()