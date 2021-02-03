#!/usr/bin/env python3
""" This script trains a e2e transformer with lfmmi. """
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>

import os
import sys
import yaml
import argparse
import subprocess
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn

import pkwrap

from lfmmi.tracking_utils import run_validation_stats
from parsers.exp_config import add_exp_options, \
    print_exp_options
from parsers.downstream_config import add_downstream_options, \
    print_downstream_options


parser = argparse.ArgumentParser(
    description="Training e2e LF-MMI model in PyTorch using pkwrap"
)
parser.add_argument(
    '--model',
    default="",
    required=True,
    help="Path to the model training file"
)
parser = add_exp_options(parser)
parser = add_downstream_options(parser)


def run_train_job(
    job_cmd,
    job_id,
    model_file,
    exp_dir,
    config,
    upconfig,
    ckpt,
    cegs_indx,
    learning_rate,
    iter_no
):
    """
        sub a single job and let ThreadPoolExecutor monitor its progress
    """
    log_file = "{}/log/train.{}.{}.log".format(exp_dir, iter_no, job_id)
    new_model = os.path.join(exp_dir, "{}.{}.pt".format(iter_no, job_id))
    job_cmp_split = job_cmd.split()
    process_out = subprocess.run([
            *job_cmp_split,
            log_file,
            model_file,
            '--mode', 'training',
            '--cegs_indx', str(cegs_indx + 1),
            '--train_stage', str(iter_no),
            '--exp_dir', str(exp_dir),
            '--lr_downstream', str(learning_rate),
            '--config', str(config),
            '--upconfig', str(upconfig),
            '--ckpt', str(ckpt),
            '--new_model', new_model
        ])
    return process_out.returncode


def merge(
    job_cmd,
    model_file,
    exp_dir,
    config,
    upconfig,
    ckpt,
    iter_no,
    num_jobs
):
    """
        Merge trained models by averaging
    """
    model_list = []
    for job_id in range(1, num_jobs+1):
        model_name = os.path.join(
            exp_dir,
            "{}.{}.pt".format(iter_no, job_id)
        )
        model_list.append(model_name)

    average_model = os.path.join(
        exp_dir, "{}.pt".format(iter_no + 1)
    )
    log_file = "{}/log/merge.{}.log".format(exp_dir, iter_no + 1)
    process_out = subprocess.run(
        [
            *job_cmd.split(),
            log_file,
            model_file,
            "--exp_dir", exp_dir,
            "--mode", "merge",
            "--new_model", average_model,
            "--merge_models", ",".join(model_list),
            "--config", str(config),
            "--upconfig", str(upconfig),
            "--ckpt", str(ckpt)
        ]
    )
    if process_out.returncode != 0:
        quit(process_out.returncode)
    else:
        for mdl in model_list:
            subprocess.run(["rm", mdl])
    if process_out.returncode != 0:
        quit(process_out.returncode)

    return


def get_exp_parameters(asr_config_path):
    config = yaml.load(open(asr_config_path, 'r'), Loader=yaml.FullLoader)
    epochs = int(config['trainer']['epochs'])
    njobs_initial = int(config['trainer']['njobs_initial'])
    njobs_final = int(config['trainer']['njobs_final'])
    lr_initial = float(config['optimizer']['lr_initial'])
    lr_final = float(config['optimizer']['lr_final'])
    return [epochs, njobs_initial, njobs_final, lr_initial, lr_final]


def print_info(narchives, niters, njobs_start, njobs_end, train_stage):
    # we start training with
    sys.stderr.write("num_archives = {}\n".format(narchives))
    sys.stderr.write("num_iteraions = {}\n".format(niters))
    sys.stderr.write("njobs_initial = {}\n".format(njobs_start))
    sys.stderr.write("njobs_final = {}\n".format(njobs_end))
    sys.stderr.write("Train Stage = {} \n".format(train_stage))


def get_cmds():
    cfg_parse = configparser.ConfigParser()
    cfg_parse.read("config")
    cmd = cfg_parse["cmd"]

    cpu_cmd, cuda_cmd = cmd["cpu_cmd"], cmd["cuda_cmd"]
    return cmd, cpu_cmd, cuda_cmd


if __name__ == "__main__":
    args = parser.parse_args()
    print_exp_options(args)
    print_downstream_options(args)

    cmd, cpu_cmd, cuda_cmd = get_cmds()

    model_file = args.model
    epochs, njobs_initial, njobs_final, \
        lr_initial, lr_final = get_exp_parameters(args.config)

    egs_dir = os.path.join(args.exp_dir, "egs")
    num_archives = pkwrap.script_utils.get_egs_info(egs_dir)
    num_archives_to_process = num_archives * epochs * 3
    num_iters = (num_archives_to_process * 2) // (njobs_initial + njobs_final)

    train_stage = args.train_stage
    assert train_stage >= 0
    print_info(
        num_archives, num_iters,
        njobs_initial, njobs_final,
        train_stage
    )

    num_archives_processed = 0
    coef = pow((lr_final / lr_initial), 1./num_iters)
    for iter_no in range(0, num_iters):
        num_jobs = pkwrap.script_utils.get_current_num_jobs(
            iter_no, num_iters, njobs_initial, 1, njobs_final
        )
        learning_rate = pow(coef, iter_no) * lr_initial

        if iter_no < train_stage:
            num_archives_processed += num_jobs
            continue
        assert (num_jobs > 0)

        sys.stderr.write("Iter = {} of {}\n".format(iter_no, num_iters))
        sys.stderr.flush()

        # Train parallelly njobs
        with ThreadPoolExecutor(max_workers=num_jobs) as executor:
            job_pool = []
            sys.stderr.write("Num jobs = {}\n".format(num_jobs))
            sys.stderr.flush()
            for job_id in range(1, num_jobs+1):
                cegs_indx = num_archives_processed % num_archives
                p = executor.submit(
                    run_train_job, cuda_cmd, job_id, model_file,
                    args.exp_dir, args.config,
                    args.upconfig, args.ckpt,
                    cegs_indx, learning_rate,
                    iter_no
                )
                num_archives_processed += 1
                job_pool.append(p)
            for p in as_completed(job_pool):
                if p.result() != 0:
                    quit(p.result())

        # Take average of the weights
        merge(
            cuda_cmd, model_file, args.exp_dir,
            args.config, args.upconfig, args.ckpt,
            iter_no, num_jobs
        )

        # Get validation statistics
        run_validation_stats(
            cuda_cmd, model_file, args.exp_dir,
            args.config, args.upconfig, args.ckpt,
            iter_no, args.graph_dir, decode_every=10,
            decode_after=20
        )
