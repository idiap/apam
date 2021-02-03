""" LFMMI training loss tracking utilities """
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>

import subprocess
import os
import sys


def compute_wer(graph_dir, data_dir, decode_dir):
    process_out = subprocess.run(
        [
            'local/score.sh',
            '--min_lmwt', '10',
            '--max_lmwt', '10',
            data_dir,
            graph_dir,
            decode_dir
        ]
    )
    if process_out.returncode != 0:
        quit(process_out.returncode)

    subprocess.run(["rm", "{}/lat.1.gz".format(decode_dir)])
    subprocess.run(["rm", "-rf", "{}/scoring".format(decode_dir)])

    wer = get_wer(os.path.join(decode_dir, 'wer_10_0.0'))
    return wer


def get_wer(fname):
    fread = open(fname, 'r')
    for line in fread:
        if '%WER' in line:
            wer = float(line.strip().split(' ')[1])
    return wer


def get_loss(fname):
    fread = open(fname, 'r')
    for line in fread:
        if "Overall" in line:
            objf = float(line.strip().split(' ')[2])
    return objf


def decode_validate(
    job_cmd,
    model_file,
    exp_dir,
    config,
    upconfig,
    ckpt,
    iter_no,
    graph_dir
):
    output_fol = "{}/decode_valid/{}_cfull".format(exp_dir, iter_no)
    log_file = "{}/decode.log".format(output_fol)

    process_out = subprocess.run(
        [
            *job_cmd.split(),
            log_file,
            model_file,
            '--exp_dir', str(exp_dir),
            '--mode', 'validation',
            '--decode_gpu', 'True',
            '--decode_validation', 'True',
            '--validation_model', str(iter_no),
            '--config', str(config),
            '--upconfig', str(upconfig),
            '--ckpt', str(ckpt),
            '|',
            'shutil/decode/latgen-faster-mapped.sh',
            '{}/words.txt'.format(graph_dir),
            os.path.join(exp_dir, '0.trans_mdl'),
            '{}/HCLG.fst'.format(graph_dir),
            os.path.join(output_fol, 'lat.1.gz')
        ]
    )

    return process_out.returncode


def track_loss(
    job_cmd,
    model_file,
    exp_dir,
    config,
    upconfig,
    ckpt,
    iter_no,
    diagnostic
):
    output_fol = "{}/decode_valid/{}_cfull".format(exp_dir, iter_no)
    log_file = "{}/{}_loss.log".format(output_fol, diagnostic)
    process_out = subprocess.run(
        [
            *job_cmd.split(),
            log_file,
            model_file,
            '--exp_dir', str(exp_dir),
            '--mode', 'validation',
            '--decode_gpu', 'True',
            '--loss_validation', 'True',
            '--diagnostic', diagnostic,
            '--validation_model', str(iter_no),
            '--config', str(config),
            '--upconfig', str(upconfig),
            '--ckpt', str(ckpt)
        ]
    )

    return process_out.returncode


def run_validation_stats(
    job_cmd,
    model_file,
    exp_dir,
    config,
    upconfig,
    ckpt,
    iter_no,
    graph_dir,
    decode_every=10,
    decode_after=20
):
    loss_result = track_loss(
        job_cmd, model_file, exp_dir,
        config, upconfig, ckpt,
        iter_no, 'valid'
    )
    if loss_result != 0:
        quit(loss_result)

    train_loss_result = track_loss(
        job_cmd, model_file, exp_dir,
        config, upconfig, ckpt,
        iter_no, 'train'
    )
    if train_loss_result != 0:
        quit(train_loss_result)

    output_fol = "{}/decode_valid/{}_cfull".format(exp_dir, iter_no)
    log_file = "{}/valid_loss.log".format(output_fol)
    objf = get_loss(log_file)
    sys.stderr.write("Iter: {} Validation Objf: {} \n".format(iter_no, objf))
    train_log_file = "{}/train_loss.log".format(output_fol)
    train_objf = get_loss(train_log_file)
    sys.stderr.write("Iter: {} Train Diagnostic Objf: {} \n".format(
                        iter_no, train_objf))
    sys.stderr.flush()

    if (iter_no % decode_every == 0) and (iter_no >= decode_after):
        decode_result = decode_validate(
            job_cmd, model_file, exp_dir,
            config, upconfig, ckpt,
            iter_no, graph_dir
        )
        if decode_result != 0:
            quit(decode_result)

        decode_dir = "{}/decode_valid/{}_cfull".format(exp_dir, iter_no)
        data_dir = "{}/egs/valid_diagnostic".format(exp_dir)
        wer = compute_wer(graph_dir, data_dir, decode_dir)
        sys.stderr.write("Iter: {} Validation WER: {} \n".format(iter_no, wer))
        sys.stderr.flush()
