from heapq import merge
from numpy import delete
import torch
from argparse import Namespace

from torch.utils.data.dataloader import DataLoader
from diffusion import gaussian_diffusion as gd
from diffusion.respace import create_spaced_diffusion, space_timesteps
from utils.parser_util import get_cond_mode
from utils.loss_util import LossType, get_main_loss

from data_loaders.humanml_utils import HML_EE_JOINT_NAMES
from model import MDM, KineMIC

from diffusion import GaussianDiffusion, GaussianDualDiffusion
from model.motion_encoders import MotionEncoderAttentionBiGRU

from typing import (
    Dict, Optional, Union, List
)

def print_model_structure(module, indent=0, name='(root)', filter="", hide_clip=False):
    prefix = '  ' * indent
    classname = module.__class__.__name__
    print(f"{prefix}{name}: {classname}")
    for pname, param in module.named_parameters(recurse=False):
        if filter in pname:
            print(f"{prefix}  ↳ param: {pname:30s} | shape={tuple(param.shape)} | requires_grad={param.requires_grad}")
    for child_name, child in module.named_children():
        if hide_clip and child_name == 'clip_model':
            print(f"{prefix}  {child_name}: {child.__class__.__name__} [HIDDEN]")
            continue
        print_model_structure(child, indent + 1, name=child_name, filter=filter, hide_clip=hide_clip)


def load_model_wo_clip(
    model : MDM,
    state_dict : Dict[str, torch.Tensor],
    from_peft_checkpoint : bool = False,
    stream: Optional[str] = None,
    peft : List[Namespace] = [],
):
    def clean_state_dict(state_dict, keys_to_delete):
        delete_list = []
        for k in state_dict.keys():
            for dk in keys_to_delete:
                if dk in k:
                    delete_list.append(k)
        for k in delete_list:
            del state_dict[k]

    #del state_dict['sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    #del state_dict['embed_timestep.sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models     
    clean_state_dict(state_dict, ['sequence_pos_encoder.pe'])
    unmatched_keys = []
    requires_peft = len(peft) > 0

    if from_peft_checkpoint:
        # the state_dict contains adapters, set them up before loading weights
        print(f"Found adapters in state_dict, plugging them before loading weights...")
        set_adapters(model, peft, stream)
        requires_peft = False # to avoid double setup of adapters

    # Load
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded [{type(model)}] model weights !")

    def clean_key_list(keys, substrings):
        return [k for k in keys if not any(substring in k for substring in substrings)]
    ignore_keys = ['clip_model', 'sequence_pos_encoder']
    missing_keys = clean_key_list(missing_keys, ignore_keys)
    unexpected_keys = clean_key_list(unexpected_keys, ignore_keys)
    unmatched_keys = clean_key_list(unmatched_keys, ignore_keys)

    if len(missing_keys) > 0 or len(unexpected_keys) > 0 or len(unmatched_keys) > 0:
        print(f"WARNING: some keys were not matched")
        print(f"\t this might be intentional if for example a stream uses Action conditioning but pre-training was done with Text conditioning.")
        print(f"\t Missing : {missing_keys}")
        print(f"\t Unexpected : {unexpected_keys}")
        print(f"\t Unmatched : {unmatched_keys}")
    
    if requires_peft:
        # Some adapters are requested, set them up
        set_adapters(model, peft, stream)


def create_model_and_diffusion(args, data, active_streams, force_single_stream=None):
    
    model_args_dict = {}
    for stream in active_streams:
        model_args_dict[stream] = get_model_args(args, getattr(args, stream), data[stream])

    if args.model_type == 'MDM':
        assert len(active_streams) == 1, "MDM model can be used only with a single stream"
        model = MDM(**model_args_dict[active_streams[0]])
    elif args.model_type == 'KineMIC':
        assert len(active_streams) == 2, "KineMIC model requires 2 active streams"
        if force_single_stream is None:
            # 2 streams are active, using both
            model = KineMIC(model_args_dict, stream_names=active_streams, tau=args.tau, judge=args.judge)
        else:
            # 2 streams are active, but it's requested to sample from a single stream
            model = MDM(**model_args_dict[force_single_stream])
    else:
        raise ValueError("{} is invalid for force_single_stream".format(force_single_stream))
    
    return model, create_gaussian_diffusion(args, data)

def get_model_args(
        args: Namespace,
        data_stream_args: Namespace,
        data : DataLoader,
    ):
    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(data_stream_args)
    num_actions = getattr(data.dataset, 'num_actions', 1)

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6
    all_goal_joint_names = []

    if data_stream_args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
        all_goal_joint_names = ['pelvis'] + HML_EE_JOINT_NAMES
    if data_stream_args.dataset in ['ntu60', 'ntu120', 'ntu-vibe']:
        if data_stream_args.data_rep == 'hml_vec':
            njoints = 263
            nfeats = 1
        elif data_stream_args.data_rep == 'xyz':
            njoints = 66
            nfeats = 1
        all_goal_joint_names = ['pelvis'] + HML_EE_JOINT_NAMES
    elif data_stream_args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    # Compatibility with old models
    if not hasattr(args, 'pred_len'):
        args.pred_len = 0
        args.context_len = 0
    
    emb_policy = args.__dict__.get('emb_policy', 'add')
    multi_target_cond = args.__dict__.get('multi_target_cond', False)
    multi_encoder_type = args.__dict__.get('multi_encoder_type', 'multi')
    target_enc_layers = args.__dict__.get('target_enc_layers', 1)

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.prior.dataset,
            'text_encoder_type': args.text_encoder_type,
            'pos_embed_max_len': args.pos_embed_max_len, 'mask_frames': args.mask_frames,
            'pred_len': args.pred_len, 'context_len': args.context_len, 'emb_policy': emb_policy,
            'all_goal_joint_names': all_goal_joint_names, 'multi_target_cond': multi_target_cond, 'multi_encoder_type': multi_encoder_type, 'target_enc_layers': target_enc_layers
            }


def create_gaussian_diffusion(args, data):
    
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False
    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)    
    loss_type =  get_main_loss('MSE')

    # Coll to different variations of GaussianDiffusion
    if args.model_type == 'MDM':
        BaseDiffusion = GaussianDiffusion
    elif args.model_type == 'KineMIC':
        BaseDiffusion = GaussianDualDiffusion
    else :
        raise ValueError(f"Invalid model_type {args.model_type}")
    print(f"\n> Using {BaseDiffusion.__name__} for model {args.model_type}")

    if not timestep_respacing:
        timestep_respacing = [steps]
    
    return create_spaced_diffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        BaseDiffusion=BaseDiffusion,
        data=data,
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambd = args.lambd,
        stream_warmup_steps = args.stream_warmup_steps,
        bank_limit=args.bank_limit,
        pretrain_diffusion_steps=args.pretrain_diffusion_steps,
        dww = args.dww,
    )

def load_saved_model(model, model_path, active_streams):
    
    # Load the checkpoint
    state_dict = torch.load(model_path, map_location='cpu')
    # Gather info
    checkpoint_info = state_dict.get('info', {})
    model_type = checkpoint_info.get('model_type', 'MDM')
    peft = checkpoint_info.get('peft', [])
    pretrain = checkpoint_info.get('pretrain', {})
    from_peft_checkpoint = len(peft) > 0

    def extract_model_from_dict(sd):
        # Extract average model when possible
        if 'model_avg' in sd.keys():
            print('loading from [avg model]')
            sd = sd['model_avg']
        else:
            if 'model' in sd:
                print('loading from [model]')
                sd = sd['model']
            else:
                print('checkpoint has no avg model, loading as usual.')
        return sd
    
    def merge_state_dicts(pretrain_sd, current_sd):
        # NOTE: the current_sd contains only the trainable components of the checkpoint
        # so we get the missing weights from the pretraining weights
        merged_sd = current_sd
        for k, v in pretrain_sd.items():
            if k not in current_sd.keys(): 
                merged_sd[k] = v # use the pretraining weights for missing keys
        return merged_sd
                

    if model_type == 'MDM':
        # just load the weights
        assert isinstance(model, MDM), 'MDM model can be loaded only with MDM class'
        assert len(active_streams) == 1, "MDM model can be used only with a single stream"
        # Load pretraining weights if available
        pretrain_state_dict = {}
        if len(pretrain) > 0 :
            pretrain_check = pretrain.get(active_streams[0], '')
            print(f"Pretrain weights loading from {pretrain_check}...")
            if pretrain_check != '':
                pretrain_state_dict = extract_model_from_dict(torch.load(pretrain_check, map_location='cpu'))
        
        print("MDM weights loading...")
        mdm_trainable_state_dict = extract_model_from_dict(state_dict)
        
        print("Merging pretrain and trainable weights...")
        mdm_state_dict = merge_state_dicts(
            pretrain_state_dict, mdm_trainable_state_dict
        )
        # .
        load_model_wo_clip(
            model, mdm_state_dict,
            from_peft_checkpoint=from_peft_checkpoint, 
            peft=peft
        )

    elif model_type == 'KineMIC':
        assert isinstance(model, KineMIC), 'KineMIC model can be loaded only with KineMIC class'
        assert len(active_streams) == 2, "KineMIC model requires 2 active streams"
        assert len(pretrain) > 0, "KineMIC model requires pretraining weights for both streams"

        # Gather the pretraining weights for each stream
        pretrain_state_dict = {}
        for stream in model.stream_names:
            print(f"Pretrain weights [{stream}] loading from {pretrain[stream]}...")
            sd = extract_model_from_dict(torch.load(pretrain[stream], map_location='cpu'))
            stream_prefix = f'streams.{stream}.'
            for k, v in sd.items():
                if k.startswith(stream_prefix):
                    pretrain_state_dict[k] = v # direct copy
                else:
                    pretrain_state_dict[stream_prefix + k] = v # add stream prefix
        
        print("KineMIC weights loading...")
        kinemic_trainable_state_dict = extract_model_from_dict(state_dict)
        
        print("Merging pretrain and trainable weights...")
        kinemic_state_dict = merge_state_dicts(
            pretrain_state_dict, kinemic_trainable_state_dict
        )
        # Load
        load_model_wo_clip(
            model, kinemic_state_dict,
            from_peft_checkpoint=from_peft_checkpoint and stream == 'target',
            peft=peft
        )

    return model

def set_adapters(model, peft:List[Namespace], stream):
    if stream in [None, 'target']:
        if isinstance(model, KineMIC):
            # NOTE: adapters are applied only to the target stream
            model.streams['target'].plug_adapters(peft) # type: ignore
        else:
            model.plug_adapters(peft)

def set_model_grads(model, grad_cfg:Namespace, model_type:str):
    if isinstance(model, KineMIC):
        for stream in model.stream_names:
            model.streams[stream].freeze(grad_cfg.frozen, model_type, stream=stream) # type: ignore
        model.share_common_parameters()  # share common parameters across streams) 
    else:
        model.freeze(grad_cfg.frozen, model_type, stream=None)