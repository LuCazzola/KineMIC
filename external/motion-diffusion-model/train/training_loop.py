
from joblib.externals.cloudpickle import k
import torch
import os
import copy
import functools
import time
import re
import math
import blobfile as bf

from typing import Optional
from tqdm import tqdm
from os.path import join as pjoin
from copy import deepcopy
from argparse import Namespace
from torch.optim import AdamW

from model.wrappers import wrap_w_classifier_free_sampling

from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from diffusion.resample import create_named_schedule_sampler
from utils import dist_util
from utils.model_util import load_model_wo_clip, set_model_grads, print_model_structure
from eval import eval_ntu
from eval.evaluators.evaluator_wrapper import EvaluatorWrapper
from data_loaders.get_data import get_single_stream_dataloader
from data_loaders.unified.scripts.motion_process import get_target_location, sample_goal, get_allowed_joint_options
from sample.generate import main as generate

# Typing imports
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from model import MDM, KineMIC
from diffusion import SpacedGaussianDiffusion
from train.train_platforms import WandBPlatform, NoPlatform

from data_loaders.tensors import unwrap_batch

from typing import (
    Dict,
    Optional,
    Union,
    List
)

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(
            self,
            args: Namespace,
            train_platform: Union[NoPlatform, WandBPlatform],
            model: Union[MDM, KineMIC],
            diffusion: SpacedGaussianDiffusion, # type: ignore
            data: DataLoader,
            eval_stream: str = 'target',
        ):
        
        self.args = args
        self.save_dir = args.save_dir
        self.train_platform = train_platform
        self.data = data

        # Model
        self.model = model
        self.model_avg = None
        # Define Validation model (and EMA if needed)
        if self.args.use_ema:
            self.model_avg = deepcopy(self.model)
        self.model_for_eval =  self.model_avg if self.args.use_ema else self.model
        if args.gen_guidance_param != 1:
            self.model_for_eval = wrap_w_classifier_free_sampling(self.model_for_eval)

        # Pointers to arguments associated to Data streams
        self.eval_stream = eval_stream # the validation data
        self.active_data_streams = [self.eval_stream] if args.model_type in ['MDM'] else args.stream_names # the training data streams

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        # Common parameters
        self.diffusion = diffusion
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.starting_checkpoint = args.starting_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps

        num_batches_per_epoch = max(1, len(self.data.dataset) // self.batch_size)
        self.num_epochs = math.ceil(self.num_steps / num_batches_per_epoch)
        self.sync_cuda = torch.cuda.is_available()
        
        # Setup training streams (adapters, gradients, etc...)
        if not self.args.learn_from_scratch:
            loaded_checkpoint = self._load_and_sync_parameters()
            assert not len(self.args.peft) > 0 or loaded_checkpoint, "When using adapters, a checkpoint must be specified."
            print(f"Adapters : {self.args.peft}]")
            set_model_grads(self.model, self.args.grad_cfg, self.args.model_type)
        else:
            print("WARNING: Training from scratch, no adapters, all gradients active.")


        # re-load the model to the device
        self.model.to(self.device)
        if self.model_avg and self.args.use_ema:
            self.model_avg.to(self.device)

        # sets up Optimizier, mixed precision, etc...
        self._setup_trainers()
        if self.resume_step and not self.args.learn_from_scratch:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
        else:
            print("WARNING: No optimizer checkpoint was loaded.")

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = self.build_eval_components(args, diffusion)

        self.use_ddp = False
        self.ddp_model = self.model

        print('\n[Trainable] params: %.3fM\n' % (sum(p.numel() for p in model.parameters_wo_clip() if p.requires_grad) / 1000000.0))

    def _setup_trainers(self):

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        if self.args.use_ema:
            self.opt = AdamW(
                # with amp, we don't need to use the mp_trainer's master_params
                (self.model.parameters() if self.use_fp16 else self.mp_trainer.master_params),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, self.args.adam_beta2),
            )
        else:
            self.opt = AdamW(
                self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
            )
        
        if self.args.model_type == 'KineMIC' and self.args.lambd.adversarial > 0.:

            if getattr(self.model, 'semantic_judge', None) is not None:
                self.diffusion.judge_opt['semantic'] = AdamW(
                    self.model.semantic_judge.parameters(),
                    lr= (self.lr * self.args.lr_judge_ratio),
                    weight_decay=self.weight_decay,
                    betas=(0.9, self.args.adam_beta2),
                )
            
            if getattr(self.model, 'style_judge', None) is not None:
                self.diffusion.judge_opt['style'] = AdamW(
                    self.model.style_judge.parameters(),
                    lr=(self.lr * self.args.lr_judge_ratio),
                    weight_decay=self.weight_decay,
                    betas=(0.9, self.args.adam_beta2),
                )

    def _load_and_sync_parameters(self):
        """
        Synch model parameters given a checkpoint / pretraining starting.
        """
        
        checkpoint = {stream : getattr(self.args, stream).checkpoint for stream in self.active_data_streams}        
        from_checkpoint = any(checkpoint.values())

        if from_checkpoint:
            # we add 1 because self.resume_step has already been done and we don't want to run it again
            # in particular we don't want to run the evaluation and generation again
            self.step += 1  
            
            for stream in self.active_data_streams:
                resume_checkpoint = checkpoint[stream]
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint) 
                logger.log(f"loading [{stream}] model from checkpoint: {resume_checkpoint}...")
                state_dict = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
                
                # NOTE: this specifies which PEFT adapters to use.
                # After pre-training weights are loaded, the adapters
                # will be put on top of the model
                peft = [getattr(self.args, name) for name in self.args.peft]

                current_model = self.model if isinstance(self.model, MDM) else self.model.streams[stream]
                current_avg_model = self.model_avg if (self.model_avg and isinstance(self.model_avg, MDM)) else (self.model_avg.streams[stream] if self.model_avg else None)
                
                if 'model_avg' in state_dict:
                    print('loading both model and model_avg')
                    if self.args.use_ema and self.model_avg : # we're using ema and we have a model_avg to load to
                        load_model_wo_clip(current_model, state_dict['model'], stream=stream, peft=peft)
                        load_model_wo_clip(current_avg_model, state_dict['model_avg'], stream=stream, peft=peft)
                    else : # use model_avg weights to initialize the model
                        load_model_wo_clip(current_model, state_dict['model_avg'], stream=stream, peft=peft)
                else:
                    state_dict = state_dict['model'] if 'model' in state_dict else state_dict
                    load_model_wo_clip(current_model, state_dict, stream=stream, peft=peft)
                    if self.model_avg and self.args.use_ema:
                        # in case we load from a legacy checkpoint, just copy the model
                        print('loading model_avg from model')
                        self.model_avg.load_state_dict(deepcopy(state_dict), strict=False)

        if any(checkpoint.values()):
            self.resume_step = 0
            self.step -= 1
        
        return from_checkpoint
        
    def _load_optimizer_state(self):
        """
        In the case of resuming training, loads the optimizer state
        """
        raise NotImplementedError("Resuming from a specific step is not implemented yet.")
        main_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )

            if self.use_fp16:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                else:
                    # load grad scaler state
                    self.scaler.load_state_dict(state_dict['scaler'])
                    # for the rest
                    state_dict = state_dict['opt']

            tgt_wd = self.opt.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt.load_state_dict(state_dict)
            print('loaded weight decay (will be replaced):', self.opt.param_groups[0]['weight_decay'])
            # preserve the weight decay parameter
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd
            self.opt.param_groups[0]['capturable'] = True                

    def build_eval_components(self, args, diffusion):
        """
        Build evaluation components for the training loop.
        note that this is applied only to the 
        """
        eval_stream_args = getattr(self.args, self.eval_stream)
        assert eval_stream_args.dataset in ['ntu60', 'ntu120', 'ntu-vibe'], "Dataset {} is not supported for evaluation.".format(eval_stream_args.dataset)

        if not args.eval_during_training : # no evaluation during training
            return None, None, None
       
        # FIXME: hardcoded for ST-GCN evaluation on NTU60 / NTU120 (for the moment)
        evaluator = {'gt' : 'stgcn', 'gen' : None}
        data_rep = {'gt': 'xyz', 'gen': eval_stream_args.data_rep}

        # GT Dataloader
        print(f'\n=== building GT dataloader [{self.eval_stream}] ===')
        eval_gt_data = get_single_stream_dataloader(
            data_stream_args=eval_stream_args,
            batch_size=args.eval_batch_size,
            split=args.eval_split, hml_mode='gt',
            evaluator=evaluator['gt'], data_rep=data_rep['gt'],
            device=dist_util.dev()
        )

        # Dataloader to generate samples
        print(f'\n=== building GEN dataloader [{self.eval_stream}] ===')
        gen_loader = get_single_stream_dataloader(
            data_stream_args=eval_stream_args,
            batch_size=args.eval_batch_size,
            split=args.eval_split, hml_mode='eval',
            evaluator=evaluator['gen'], data_rep=data_rep['gen'],
            device=dist_util.dev()
        )
        
        mm_num_samples, mm_num_repeats = 0, 0  # mm is super slow hence we won't run it during training
        
        eval_data = {
            'test': lambda: eval_ntu.get_mdm_loader(self.args,
                self.model_for_eval, diffusion, args.eval_batch_size,
                gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                args.eval_num_samples, scale=args.gen_guidance_param, cond_mode=eval_stream_args.cond_mode
            )
        }
        
        eval_wrapper = EvaluatorWrapper(eval_stream_args.dataset, dist_util.dev(), task_split=eval_stream_args.task_split, fewshot_id=eval_stream_args.fewshot_id)
        
        return eval_wrapper, eval_data, eval_gt_data

    def run_loop(self, eval_stream='target'):
        """
        The training loop.
        Supports both single-stream (MDM) and two-stream (Cycle-MDM) training.
        """

        for ep in tqdm(range(self.num_epochs), desc='Epoch'):
            for batch in self.data:
                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break
                
                # to handle different datasets gettitems 
                motion, cond = unwrap_batch(batch, self.data.dataset, self.active_data_streams)

                # retrieve current batch size
                bs = motion[self.active_data_streams[0]].shape[0]
                if len(self.active_data_streams) == 2: # Truncate to smaller batch size (if needed)
                    bs = min(motion[self.active_data_streams[0]].shape[0], motion[self.active_data_streams[1]].shape[0])

                # Move to device
                for stream in self.active_data_streams:
                    motion[stream] = motion[stream].to(self.device)
                    cond[stream]['y'] = {key: val[:bs].to(self.device) if torch.is_tensor(val) else val for key, val in cond[stream]['y'].items()}
                
                self.diffusion.iteration = self.total_step() # make diffusion module aware of current step
                
                # Training step
                self.run_step(motion, cond)

                # Logging
                if self.total_step() % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(), group_name='Loss')
                
                # Evaluation and saving
                if self.total_step() % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    if self.args.use_ema:
                        self.model_avg.eval()
                    self.generate_during_training(eval_stream, 'single')
                    self.evaluate(eval_stream) 
                    self.model.train()
                    if self.args.use_ema:
                        self.model_avg.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.total_step() > 0:
                        return
                self.step += 1

            if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                break

        # Save the last checkpoint if it wasn't already saved.
        if (self.total_step() - 1) % self.save_interval != 0:
            self.save()
            self.evaluate(eval_stream)

    def evaluate(self, stream: str):
        """
        Perform evaluation for the given Dataset
        """
        if not self.args.eval_during_training:
            return
        assert stream == 'target', "Only target stream evaluation is supported."
        
        stream_args = getattr(self.args, stream)
        start_eval = time.time()
        assert self.eval_wrapper is not None, "Evaluation for datasets {} is not supported.".format(stream_args.dataset)
        
        print('\n>>> Running evaluation loop <<<\n')
        log_file = os.path.join(self.save_dir, f'eval_{stream_args.dataset}_{(self.total_step()):09d}.log')
        diversity_times = 300
        mm_num_times = 0  # mm is super slow hence we won't run it during training
        
        assert stream_args.dataset in ['ntu60', 'ntu120', 'ntu-vibe'], "Only NTU60 and NTU120 evaluation is supported during training."
        eval_dict = eval_ntu.evaluation(
            self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
            replication_times=self.args.eval_rep_times,
            diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False
        )
        print("Eval. results -> ", eval_dict)
        
        for k, v in eval_dict.items():
            if k.startswith('R_precision') or k.startswith('Accuracy'):
                for i in range(len(v)):
                    self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                        iteration=self.total_step(),
                                                        group_name='Eval')
            else:
                self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(),
                                                    group_name='Eval')
        
        end_eval = time.time()
        print(f'Evaluation time: {(round(end_eval-start_eval)/60):.2f} min.')

    def _clip_generator_grad(self):
        if self.args.grad_clip > 0. and self.args.stream_warmup_steps <= self.total_step():
            if self.args.model_type == 'MDM':
                # clip all the model gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)
            else:
                from itertools import chain
                trainable_params = [
                    p for p in 
                    chain(self.model.streams['target'].parameters(), self.model.mic_module.parameters()) 
                    if p.requires_grad
                ]
                # clip the norm of the TOTAL gradient once.
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.args.grad_clip)


    def run_step(self,
        batch: Dict[str, Tensor],
        cond: Dict[str, Dict[str, Tensor]],
    ):
        self.forward_backward(batch, cond)
        self._clip_generator_grad()
        self.mp_trainer.optimize(self.opt)
        self.update_average_model()
        self._anneal_lr()
        self.log_step()

    def update_average_model(self):
        # update the average model using exponential moving average
        if self.args.use_ema:
            # master params are FP32
            params = self.model.parameters() if self.use_fp16 else self.mp_trainer.master_params
            for param, avg_param in zip(params, self.model_avg.parameters()):
                avg_param.data.mul_(self.args.avg_model_beta).add_(param.data, alpha=(1 - self.args.avg_model_beta))

    def forward_backward(
        self,
        batch: Dict[str, Tensor],
        cond: Dict[str, Dict[str, Tensor]],
    ):
        """
        Computes model's forward and backward pass.
        """
        first_stream = self.active_data_streams[0] # always guaranteed to be present
        curr_batch_size = batch[first_stream].shape[0]
        
        if len(self.active_data_streams) == 1:
            # Single stream, remove [stream] key
            batch, cond = batch[first_stream], cond[first_stream]
        elif len(self.active_data_streams) == 2:
            assert batch[self.active_data_streams[0]].shape == batch[self.active_data_streams[1]].shape, "Sanity check"

        self.mp_trainer.zero_grad()
        for i in range(0, curr_batch_size, self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= curr_batch_size
            t, weights = self.schedule_sampler.sample(curr_batch_size, dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                dataset=self.data.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():   
                    losses = compute_losses()
            
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.total_step() / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.total_step())
        logger.logkv("samples", (self.total_step() + 1) * self.global_batch)

    def ckpt_file_name(self, stream=None):
        if stream:
            return f"model_{stream}_{(self.total_step()):09d}.pt"
        return f"model_{(self.total_step()):09d}.pt"

    def generate_during_training(
        self,
        sampling_stream: str,
        sampling_mode: str
    ):
        if not self.args.gen_during_training:
            return
        gen_args = copy.deepcopy(self.args)
        gen_args.model_path = os.path.join(self.save_dir, self.ckpt_file_name())
        gen_args.output_dir = os.path.join(self.save_dir, f'{self.ckpt_file_name()}.samples')
        gen_args.num_samples = self.args.gen_num_samples
        gen_args.num_repetitions = self.args.gen_num_repetitions
        gen_args.guidance_param = self.args.gen_guidance_param
        gen_args.motion_length = self.args.gen_motion_length
        gen_args.sampling_stream = sampling_stream
        gen_args.sampling_mode = sampling_mode
        gen_args.input_text = gen_args.text_prompt = gen_args.action_file = gen_args.action_name = gen_args.dynamic_text_path = ''
        gen_args.action_id = []
        all_sample_save_path = generate(gen_args)
        self.train_platform.report_media(title='Motion', series='Predicted Motion', iteration=self.total_step(), local_path=all_sample_save_path)        

    
    def find_resume_checkpoint(self) -> Optional[str]:
        '''look for all file in save directory in the pattent of model{number}.pt
            and return the one with the highest step number.

        TODO: Implement this function (alredy existing in MDM), so that find model will call it in case a ckpt exist.
        TODO: Change call for find_resume_checkpoint and send save_dir as arg.
        TODO: This means ignoring the flag of resume_checkpoint in case some other ckpts exists in that dir!
        '''
        matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None
    
    def total_step(self):
        return self.step + self.resume_step
    
    def save(self):
        def save_checkpoint():
            
            def prune_state_dict(state_dict_to_prune):
                
                new_state_dict = {}
                KEEP_LIST = ['semantic_judge', 'style_judge']
                IGNORE_LIST = ['streams.prior', 'clip_model']

                for name, param in state_dict_to_prune.items():                    
                    # Always ignore anything in the IGNORE_LIST.
                    if any(substring in name for substring in IGNORE_LIST):
                        continue
                    # Always keep anything in the KEEP_LIST.
                    if any(substring in name for substring in KEEP_LIST):
                        new_state_dict[name] = param
                        continue
                    # For everything else, keep it only if it's trainable.
                    try:
                        model_param = self.model.get_parameter(name)
                        if model_param.requires_grad:
                            new_state_dict[name] = param
                    except AttributeError:
                        new_state_dict[name] = param
                        
                return new_state_dict
            
            # Get the full state dict for the main model
            if self.use_fp16:
                full_state_dict = self.model.state_dict()
            else:
                full_state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)

            # format
            state_dict = {
                'model': prune_state_dict(full_state_dict),
                'info' : {
                    'model_type': self.args.model_type,
                    'peft' : [getattr(self.args, name) for name in self.args.peft],
                    'pretrain' : {s : getattr(self.args, s).checkpoint for s in self.active_data_streams},
                }
            }

            if self.args.use_ema:
                # add average model entry (if present)
                state_dict_avg = self.model_avg.state_dict()
                state_dict['model_avg'] = prune_state_dict(state_dict_avg)

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)
            logger.log(f"model saved to {bf.join(self.save_dir, filename)}")

        save_checkpoint()

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.total_step()):09d}.pt"),
            "wb",
        ) as f:
            opt_state = self.opt.state_dict()
            if self.use_fp16:
                # with fp16 we also save the state dict
                opt_state = {
                    'opt': opt_state,
                    'scaler': self.scaler.state_dict(),
                }

            torch.save(opt_state, f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)