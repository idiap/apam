""" LFMMI training utility class"""
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>

import os
import sys
import yaml

import torch
import torch.optim as optim
from torch.optim import Adam

import pkwrap
from pkwrap.script_utils import feat_writer, \
        feat_reader_gen, egs_reader_gen
from pkwrap.chain import KaldiChainObjfFunction, \
        prepare_minibatch


class LFMMITrainer():
    """Handles acoustic model training with LFMMI
    Arguments:
        exp_dir: Experiment directory with egs folder
        config_path: Config file contains parameters related to model, training
                criterion, optimizer, directory paths
        upconfig: Config to the upstream model.
        ckpt: path to the pre-trained ckpt
        get_model: get_model is the function which will take model arguments
                   from config and return the model to be used for training.
                   This will enable all different kind of model training with
                   same Trainer.
    """
    def __init__(self, exp_dir, asr_config_path, upconfig, ckpt, get_model):

        self.upconfig = upconfig
        self.ckpt = ckpt
        self.asr_config_path = asr_config_path
        config = yaml.load(open(asr_config_path, 'r'), Loader=yaml.FullLoader)
        self.get_model = get_model

        # optimizer
        self.lr_upstream = float(config['optimizer']['lr_upstream'])
        self.weight_decay_upstream = float(
            config['optimizer']['weight_decay_upstream']
        )
        self.weight_decay_downstream = float(
            config['optimizer']['weight_decay_downstream']
        )

        # Trainer
        self.minibatch_size = int(config['trainer']['minibatch_size'])
        self.grad_acc_steps = int(config['trainer']['grad_acc_steps'])

        # Criterion
        self.l2_regularize = float(config['criterion']['l2_regularize'])
        self.leaky_hmm_coefficient = float(
            config['criterion']['leaky_hmm_coefficient']
        )
        self.oor_regularize = float(config['criterion']['oor_regularize'])
        self.xent_regularize = float(config['criterion']['xent_regularize'])

        # Experiment related options
        self.dirname = exp_dir
        self.den_fst_path = os.path.join(self.dirname, "den.fst")
        self.egs_dir = os.path.join(self.dirname, 'egs')
        self.output_subsampling = int(
            config['model']['downstream']['output_subsampling']
        )

        # Upstream parameters
        self.finetune = config['model']['upstream']['fine_tune']

        self.input_dim, self.output_dim = self.get_dimensions(exp_dir)

    def get_dimensions(self, dirname):
        output_dim = None
        with open(os.path.join(dirname, "num_pdfs")) as ipf:
            output_dim = int(ipf.readline().strip())
        assert output_dim is not None
        feat_dim = None
        with open(os.path.join(dirname, "feat_dim")) as ipf:
            feat_dim = int(ipf.readline().strip())
        assert feat_dim is not None
        return feat_dim, output_dim

    def initialize_model(self):
        upstream_model, downstream_model = self.get_model(
            self.asr_config_path,
            self.upconfig,
            self.input_dim,
            self.output_dim,
            self.ckpt
        )
        return upstream_model, downstream_model

    def load_model(
        self,
        upstream_model,
        downstream_model,
        optimizer=None,
        from_path=None
    ):
        if from_path is None:
            return
        sys.stderr.write('Load model from {}'.format(from_path))
        all_states = torch.load(from_path, map_location='cpu')
        load_model_list = ['Upstream', 'Downstream', 'Optimizer']

        if 'Upstream' in load_model_list:
            try:
                upstream_model.load_state_dict(all_states['Upstream'])
                sys.stderr.write('[Upstream] - Loaded')
            except:
                sys.stderr.write('[Upstream - X]')

        if 'Downstream' in load_model_list:
            try:
                state_dict = all_states['Downstream']
                # perform load
                downstream_model.load_state_dict(state_dict)
                sys.stderr.write('[Downstream] - Loaded')
            except:
                sys.stderr.write('[Downstream - X]')

        if 'Optimizer' in load_model_list and optimizer is not None:
            try:
                optimizer.load_state_dict(all_states['Optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
                sys.stderr.write('[Optimizer] - Loaded')
            except:
                sys.stderr.write('[Optimizer - X]')

        sys.stderr.write('Model loading complete!')
        return

    def save_model(
        self,
        upstream_model,
        downstream_model,
        optimizer,
        model_path
    ):
        all_states = {
            'Upstream': upstream_model.state_dict(),
            'Downstream': downstream_model.state_dict(),
            'Optimizer': optimizer.state_dict() if optimizer is not None else None,
        }
        torch.save(all_states, model_path)
        return

    def train_lfmmi_one_iteration(
        self,
        iteration,
        cegs_indx,
        lr_downstream,
        print_interval=10,
        use_gpu=True
    ):
        if iteration == 0:
            finetune = False
        else:
            finetune = self.finetune
        training_opts = pkwrap.kaldi.chain.CreateChainTrainingOptions(
            self.l2_regularize,
            self.oor_regularize,
            self.leaky_hmm_coefficient,
            self.xent_regularize
        )

        den_graph = pkwrap.kaldi.chain.LoadDenominatorGraph(
            self.den_fst_path,
            self.output_dim
        )
        criterion = KaldiChainObjfFunction.apply
        if use_gpu:
            self.downstream = self.downstream.cuda()
            self.upstream = self.upstream.cuda()
            self.upstream.set_device('cuda')

        self.downstream.train()
        self.upstream.eval()
        param_downstream = list(self.downstream.parameters())
        optimizer_downstream = Adam(
            param_downstream,
            lr=lr_downstream,
            weight_decay=self.weight_decay_downstream
        )
        optimizer_upstream = None
        if finetune:
            self.upstream.train()
            param_upstream = list(
                filter(lambda p: p.requires_grad, self.upstream.parameters())
            )
            optimizer_upstream = Adam(
                param_upstream,
                lr=self.lr_upstream,
                weight_decay=self.weight_decay_upstream
            )

        sys.stderr.write("Iter = {}\n".format(cegs_indx))
        sys.stderr.flush()
        egs_file = "ark:{}/cegs.{}.ark".format(self.egs_dir, cegs_indx)
        acc_sum = torch.tensor(0., requires_grad=False)
        effective_print_interval = print_interval * self.grad_acc_steps
        for mb_id, merged_egs in enumerate(
            prepare_minibatch(egs_file, self.minibatch_size)
        ):
            try:
                features = pkwrap.kaldi.chain.GetFeaturesFromEgs(merged_egs)
                N, T, F = features.shape
                output, xent_output = self.downstream(self.upstream(features))
                sup = pkwrap.kaldi.chain.GetSupervisionFromEgs(merged_egs)
                deriv = criterion(
                    training_opts, den_graph, sup, output, xent_output
                )
                acc_sum.add_(deriv[0].item())
                if (mb_id > 0) and ((mb_id % effective_print_interval) == 0):
                    sys.stderr.write("Overall objf {}\n".format(
                        acc_sum / effective_print_interval)
                    )
                    acc_sum.zero_()
                if self.grad_acc_steps > 1:
                    deriv = deriv / self.grad_acc_steps
                deriv.backward()
                if (mb_id + 1) % self.grad_acc_steps == 0:
                    optimizer_downstream.step()
                    optimizer_downstream.zero_grad()
                    if finetune:
                        optimizer_upstream.step()
                        optimizer_upstream.zero_grad()

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print('CUDA out of memory at step: {}, length {}'.format(mb_id, T))
                    torch.cuda.empty_cache()
                    optimizer_downstream.zero_grad()
                    if finetune:
                        optimizer_upstream.zero_grad()
                else:
                    raise

        try:
            optimizer_downstream.step()
            optimizer_downstream.zero_grad()
            if finetune:
                optimizer_upstream.step()
                optimizer_upstream.zero_grad()
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print('CUDA out of memory at final step. Optimizer states issues sadly')
                torch.cuda.empty_cache()
                optimizer_downstream.zero_grad()
                if finetune:
                    optimizer_upstream.zero_grad()
            else:
                raise
        sys.stdout.flush()
        sys.stderr.flush()

        self.upstream = self.upstream.cpu()
        self.downstream = self.downstream.cpu()

        return

    def run_train(
        self,
        train_stage,
        lr_downstream,
        cegs_indx,
        new_model
    ):
        pkwrap.kaldi.InstantiateKaldiCuda()
        # load model
        self.upstream, self.downstream = self.initialize_model()
        base_model = os.path.join(
            self.dirname, "{}.pt".format(train_stage)
        )
        self.load_model(self.upstream, self.downstream, None, base_model)
        sys.stderr.write("Loaded base model from {}".format(base_model))
        sys.stderr.flush()
        # train model
        self.train_lfmmi_one_iteration(
            train_stage,
            cegs_indx,
            lr_downstream=lr_downstream,
            print_interval=10,
            use_gpu=True
        )
        self.save_model(
            self.upstream, self.downstream,
            None, new_model
        )
        return

    def loss_validation_set(
        self,
        diagnostic_file,
        use_gpu=True
    ):
        self.upstream.eval()
        self.downstream.eval()
        # We set regularization losses to 0
        training_opts = pkwrap.kaldi.chain.CreateChainTrainingOptions(
            0.0, 0.0,
            self.leaky_hmm_coefficient,
            0.0
        )

        den_graph = pkwrap.kaldi.chain.LoadDenominatorGraph(
            self.den_fst_path,
            self.output_dim
        )
        criterion = KaldiChainObjfFunction.apply
        if use_gpu:
            self.downstream = self.downstream.cuda()
            self.upstream = self.upstream.cuda()
            self.upstream.set_device('cuda')

        sys.stderr.write("Running {} diagnostics\n".format(diagnostic_file))
        egs_file = "ark:{}/{}_diagnostic.cegs".format(
            self.egs_dir, diagnostic_file
        )
        acc_sum = torch.tensor(0., requires_grad=False)

        total_samples = 0
        for mb_id, merged_egs in enumerate(prepare_minibatch(egs_file, 1)):
            features = pkwrap.kaldi.chain.GetFeaturesFromEgs(merged_egs)
            N, T, F = features.shape
            output, xent_output = self.downstream(self.upstream(features))
            output = output.contiguous()
            xent_output = xent_output.contiguous()
            sup = pkwrap.kaldi.chain.GetSupervisionFromEgs(merged_egs)
            deriv = criterion(
                training_opts, den_graph, sup, output, xent_output
            )
            acc_sum.add_(deriv[0])
            total_samples += 1
        sys.stderr.write("Overall objf {}\n".format(acc_sum / total_samples))
        sys.stderr.write("Total Samples {}\n".format(total_samples))
        sys.stderr.flush()
        return

    def decode_validation_set(
        self,
        decode_output,
        decode_gpu=True
    ):
        self.upstream.eval()
        self.downstream.eval()

        sys.stderr.write("Decoding validation\n")
        egs_file = "ark:{}/valid_diagnostic.cegs".format(self.egs_dir)
        writer_spec = "ark:{}".format(decode_output)
        writer = feat_writer(writer_spec)
        if decode_gpu:
            self.downstream = self.downstream.cuda()
            self.upstream = self.upstream.cuda()
            self.upstream.set_device('cuda')
        total_keys = 0
        for key, feats in egs_reader_gen(egs_file):
            total_keys += 1
            N, T, F = feats.shape
            post, _ = self.downstream(self.upstream(feats))
            post = post.squeeze(0).cpu()
            writer.Write(key, pkwrap.kaldi.matrix.TensorToKaldiMatrix(post))
            sys.stderr.write("Wrote {}\n ".format(key))
            sys.stderr.flush()
        writer.Close()

    def run_validation(
        self,
        validation_model,
        diagnostic='valid',
        loss_validation=True,
        decode_validation=False,
        decode_gpu=False,
        decode_output=None
    ):
        pkwrap.kaldi.InstantiateKaldiCuda()
        with torch.no_grad():
            # load model
            self.upstream, self.downstream = self.initialize_model()
            base_model = os.path.join(
                self.dirname, "{}.pt".format(validation_model)
            )
            try:
                self.load_model(
                    self.upstream, self.downstream, None, base_model
                )
            except:
                sys.stderr.write("Cannot load model {}".format(base_model))
                sys.stderr.flush()
                quit(1)
            sys.stderr.write("Loading base model from {}".format(base_model))
            sys.stderr.flush()

            if loss_validation:
                self.loss_validation_set(
                    diagnostic,
                    use_gpu=decode_gpu
                )
            if decode_validation:
                self.decode_validation_set(
                    decode_output,
                    decode_gpu
                )
        return

    def decode(
        self,
        train_stage,
        decode_feats,
        decode_output,
        decode_gpu=False
    ):
        with torch.no_grad():
            self.upstream, self.downstream = self.initialize_model()
            base_model = os.path.join(
                self.dirname, "{}.pt".format(train_stage)
            )
            try:
                self.load_model(
                    self.upstream, self.downstream, None, base_model
                )
            except:
                sys.stderr.write("Cannot load model {}".format(base_model))
                sys.stderr.flush()
                quit(1)
            self.upstream.eval()
            self.downstream.eval()
            writer_spec = "ark:{}".format(decode_output)
            writer = feat_writer(writer_spec)
            if decode_gpu:
                self.downstream = self.downstream.cuda()
                self.upstream = self.upstream.cuda()
            for key, feats in feat_reader_gen(decode_feats):
                feats = feats.unsqueeze(0)
                post, _ = self.downstream(self.upstream(feats))
                post = post.squeeze(0).cpu()
                writer.Write(key, pkwrap.kaldi.matrix.TensorToKaldiMatrix(post))
                sys.stderr.write("Wrote {}\n ".format(key))
                sys.stderr.flush()
            writer.Close()

    def merge(self, merge_models, new_model):
        with torch.no_grad():
            base_models = merge_models.split(',')
            sys.stderr.write(" ".join(base_models))
            sys.stderr.write("\n")
            assert len(base_models) > 0
            upstream, downstream = self.initialize_model()
            try:
                self.load_model(upstream, downstream, None, base_models[0])
            except:
                sys.stderr.write("Cannot load model {}".format(base_models[0]))
                sys.stderr.flush()
                quit(1)

            print(list(upstream.parameters())[0])
            print(list(downstream.parameters())[0])
            sys.stderr.write("\n")
            upstream_acc = dict(upstream.named_parameters())
            downstream_acc = dict(downstream.named_parameters())
            for model_name in base_models[1:]:
                this_upstream, this_downstream = self.initialize_model()
                self.load_model(
                    this_upstream, this_downstream, None, model_name
                )
                for name, params in this_upstream.named_parameters():
                    upstream_acc[name].data.add_(params.data)
                for name, params in this_downstream.named_parameters():
                    downstream_acc[name].data.add_(params.data)
            weight = 1.0/len(base_models)
            for name in upstream_acc:
                upstream_acc[name].data.mul_(weight)
            for name in downstream_acc:
                downstream_acc[name].data.mul_(weight)
            print(list(upstream_acc.keys())[0])
            print(list(downstream_acc.keys())[0])
            sys.stderr.write("\n")
            self.save_model(upstream, downstream, None, new_model)
        return
