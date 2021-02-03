""" Parser for asr training options """
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>

import argparse
import sys


def add_downstream_options(parser):
    parser.add_argument(
        '--ckpt',
        default='',
        type=str,
        help='Path to upstream pre-trained checkpoint, required if using other than baseline',
        required=True
    )
    parser.add_argument(
        '--config',
        default='config/asr-downstream.yaml',
        type=str, help='Path to downstream experiment config.',
        required=True
    )
    parser.add_argument(
        '--upconfig',
        default='default',
        type=str, help='Path to the option upstream config. Pass default to use from checkpoint',
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Disable GPU training.'
    )
    return parser


def print_downstream_options(args):
    sys.stderr.write("""
        Downstream Config:
        Checkpoint: {ckpt}
        ASR Config: {config}
        Upconfig: {upconfig}
        CPU Training: {cpu}
    """.format(**vars(args)))
