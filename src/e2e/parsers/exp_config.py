""" Parser for asr training options """
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>

import argparse
import sys


def add_exp_options(parser):
    parser.add_argument(
        "--graph_dir",
        type=str,
        default=None,
        help="Path to the decoding graph"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="Path to experiment folder name"
    )
    parser.add_argument(
        "--train_stage",
        type=int,
        default=-1,
        help="Training stage for restarting paused training"
    )
    return parser


def print_exp_options(args):
    sys.stderr.write("""
        Experiment Config:
        Decoding Graph Directory: {graph_dir}
        Experiment Directory: {exp_dir}
    """.format(**vars(args)))
