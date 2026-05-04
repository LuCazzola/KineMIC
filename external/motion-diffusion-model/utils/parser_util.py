import argparse
import os
import json
from argparse import ArgumentParser, Namespace

class Pfx():
    """A class to hold the prefixes for different argument groups and related utility functions"""
    # Prefixes
    LORA = 'LoRA'
    MOE = 'MoELoRA'
    PRIOR = 'prior'
    TARGET = 'target'
    GRAD_CFG = 'grad_cfg'
    LAMBD = 'lambd'
    JUDGE = 'judge'
    # Prefix groups
    STREAMS = [PRIOR, TARGET]
    ADAPTERS = [LORA, MOE]
    GRADIENTS = [GRAD_CFG]
    COEFF = [LAMBD]
    ADVERSARIAL = [JUDGE]
    # All combined
    ALL_PREFIXES = STREAMS + ADAPTERS + GRADIENTS + COEFF + ADVERSARIAL

    @staticmethod
    def wrap_args_by_prefix(
        args : Namespace,
    ):
        """ Wraps adapter arguments into sub-namespaces for better organization."""
        
        def group_args_by_prefix(args: Namespace, prefix: str):
            """Groups arguments by a given prefix and returns the extracted namespace and list of keys."""
            group_dict = {k[len(prefix)+1:]: v for k, v in vars(args).items() if k.startswith(f"{prefix}_")}
            extracted_keys = list(group_dict.keys())
            # Remember: extracted_keys contain keys after stripping the prefix.
            # You likely want the full keys to delete from args:
            full_keys = [f"{prefix}_{k}" for k in extracted_keys]
            return Namespace(**group_dict), full_keys
        
        for prefix in Pfx.ALL_PREFIXES:
            new_args, old_keys = group_args_by_prefix(args, prefix)
            setattr(args, prefix, new_args) 
            for k in old_keys:
                delattr(args, k)

        return args
    
    @staticmethod
    def serialize_args(ns):
        if isinstance(ns, Namespace):
            return {k: Pfx.serialize_args(v) for k, v in vars(ns).items()}
        return ns
    
    @staticmethod
    def de_serialize_args(d):
        if isinstance(d, dict):
            return Namespace(**{k: Pfx.de_serialize_args(v) for k, v in d.items()})
        return d

    @staticmethod
    def get_opts_from_adapter_name(name: str, adapter_args: Namespace):
        if name == Pfx.LORA:
            from model.adapters.loralib import namespace_to_lora_opt
            return namespace_to_lora_opt(adapter_args)
        elif name == Pfx.MOE:
            from model.adapters.moelib import namespace_to_moe_opt
            return namespace_to_moe_opt(adapter_args)
        else :
            raise ValueError(f"Unknown adapter name: {name}. Supported: [{Pfx.ADAPTERS}].")

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_peft_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion', 'peft']:        
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)        
    
    # load args from model
    if args.model_path != '':  # if not using external results file
        args = load_args_from_model(args, args_to_overwrite)

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    
    return args

def load_args_from_model(args, args_to_overwrite):
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    # Create a flattened version for easier lookup
    flattened_args = {}
    
    for key, value in model_args.items():
        if isinstance(value, dict):
            # Handle nested dictionaries (like prior, target, LoRA, MoE)
            for nested_key, nested_value in value.items():
                flattened_key = f"{key}_{nested_key}"
                flattened_args[flattened_key] = nested_value
        else:
            flattened_args[key] = value

    # Now overwrite args with values from flattened_args
    for arg_name in args_to_overwrite:
        if arg_name in flattened_args:
            setattr(args, arg_name, flattened_args[arg_name])
        #elif 'cond_mode' in model_args:  # backward compatibility
        #    unconstrained = (model_args['cond_mode'] == 'no_cond')
        #    setattr(args, 'unconstrained', unconstrained)
        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(
                arg_name, getattr(args, arg_name, 'N/A')))
    return args

def apply_rules(args):
    
    # For prefix completion
    if args.pred_len == 0:
        args.pred_len = args.context_len
    
    # Adversarial
    # . Disabling options
    if args.lambd.adversarial == 0.0:
        args.judge.type = None
    if args.judge.type is None:
        args.lambd.adversarial = 0.0
    # 
    if args.judge.reg != 'gp':
        args.lambd.gp = 0.0 # disable dradient penalty if it's not requested as regularization method.
    if len(args.judge.task) == 1:
        if 'style' in args.judge.task :
            args.lambd.adversarial_alpha = 1.0 # only style
        elif 'semantic' in args.judge.task :
            args.lambd.adversarial_alpha = 0.0 # only semantic

    # Broadcasting of the starting checkpoint to both streams
    if args.prior.checkpoint == '' and args.target.checkpoint == '' and getattr(args, 'starting_checkpoint', '') != '':
        args.prior.checkpoint = args.target.checkpoint = args.starting_checkpoint

    
    return args

def get_args_per_group_name(parser, args, group_name):
    
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())

    return ValueError('group_name was not found.')


def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('--model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'WandBPlatform'], type=str, help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--external_mode", default=False, type=bool, help="For backward cometability, do not change or delete.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str, help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=100, type=int, choices=[50, 100, 1000], help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    group.add_argument("--pretrain_diffusion_steps", default=50, type=int, choices=[50], help="Number of diffusion steps of the pretrained checkpoint (when KineMIC)")

def add_model_options(parser):
    group = parser.add_argument_group('model')
    ### NEW
    #
    group.add_argument("--model_type", default='MDM', type=str, choices=['MDM', 'KineMIC'], help="Type of the model to be used. MDM for single stream, CycleMDM for two streams.")
    group.add_argument("--stream_names", nargs='*', type=str, default=Pfx.STREAMS, choices=Pfx.STREAMS, help="Names of the streams to be used in the model. Used for CycleMDM only.")
    group.add_argument("--single_stream", default=Pfx.TARGET, type=str, choices=[Pfx.PRIOR, Pfx.TARGET], help="in the case of using MDM model, specifies which data stream should be used")
    #
    group.add_argument("--learn_from_scratch", action='store_true', help="Forces the model to IGNORE usage of adapters and specified checkpoints. (useful for training from scratch).")
    group.add_argument(f"--{Pfx.GRAD_CFG}_frozen", nargs='*', type=str, default=['norms'], choices=['transformer_attn', 'transformer_ff', 'conditioning', 'input_process', 'output_process', 'norms', 'embed_timestep'], help="Specifies which parts of the model should be forzen.")
    group.add_argument("--stream_warmup_steps", default=0, type=int, help="Number of steps in which Adversarial Loss is not applied.")
    # KineMIC coeff
    # . Main Loss coefficients
    group.add_argument(f"--{Pfx.LAMBD}_rec", default=1.0, type=float, help="Main reconstruction loss.")
    group.add_argument(f"--{Pfx.LAMBD}_window_rec", default=1.0, type=float, help="Prior Window reconstruction loss.")
    group.add_argument(f"--{Pfx.LAMBD}_window_distill", default=0.5, type=float, help="Prior Window latent distillation loss.")
    group.add_argument(f"--{Pfx.LAMBD}_contrastive", default=1.0, type=float, help="Contrastive loss.")
    group.add_argument(f"--{Pfx.LAMBD}_adversarial", default=0.0, type=float, help="Adversarial loss.")
    group.add_argument(f"--{Pfx.LAMBD}_adversarial_alpha", default=0.5, type=float, help="Balancing among the 2 adversarial losses (style and semantic, if both are used). 1 - only style, 0 - only semantic.")
    # . Semantic Alignment parameters
    group.add_argument("--top_k_sp", default=250, type=int, help="Number of nearest neighbors to use in KNN for soft positives mining")
    group.add_argument("--tau", default=0.07, type=float, help="Tau for Temperature in the contrastive loss.")
    group.add_argument("--dww", action='store_true', help="If true, will use Dynamic Window Weighting in the contrastive loss.")
    group.add_argument("--bank_limit", default=60, type=int, help="Size of the memory bank for contrastive loss.")
    # . Adversarial loss options
    group.add_argument(f"--{Pfx.JUDGE}_type", default=None, type=str, choices=[None, 'd', 'c'], help="If not None, will use an adversarial loss. d - a discriminator, c - a critic.")
    group.add_argument(f"--{Pfx.JUDGE}_reg", default=None, type=str, choices=[None, 'sn', 'gp'], help="Type of regularization for the adversarial loss. sn - spectral norm, gp - gradient penalty, None - no regularization.")
    group.add_argument(f"--{Pfx.JUDGE}_task", default=['semantic', 'style'], nargs='*', type=str, choices=['semantic', 'style'], help="Which discriminators to add (Semantic: on latent MIC rep, Style: on output projection)")
    group.add_argument(f"--{Pfx.LAMBD}_gp", default=10.0, type=float, help="Gradient penalty loss (If using a discriminator with GP reg.)")
    # MDM coeff
    # NOTE: the below coefficients are actually useless in MDM training since the hml_vec data format is used
    group.add_argument(f"--{Pfx.LAMBD}_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument(f"--{Pfx.LAMBD}_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument(f"--{Pfx.LAMBD}_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument(f"--{Pfx.LAMBD}_target_loc", default=0.0, type=float, help="For HumanML only, when . L2 with target location.")

    # .
    # Other model configurations
    group.add_argument("--arch", default='trans_enc', choices=['trans_enc', 'trans_dec', 'gru'], type=str, help="Architecture types as reported in the paper.")
    group.add_argument("--text_encoder_type", default='clip', choices=['clip', 'bert'], type=str, help="Text encoder type.")
    group.add_argument("--emb_trans_dec", action='store_true', help="For trans_dec architecture only, if true, will inject condition as a class token (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int, help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int, help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float, help="The probability of masking the condition during training. For classifier-free guidance learning.")
    group.add_argument("--mask_frames", default=True, type=bool, help="If true, will fix Rotem's bug and mask invalid frames.")
    group.add_argument("--pos_embed_max_len", default=5000, type=int, help="Pose embedding max length.")
    group.add_argument("--use_ema", action='store_true', help="If set, will use EMA model averaging.")
    group.add_argument("--multi_target_cond", action='store_true', help="If true, enable multi-target conditioning (aka Sigal's model).")
    group.add_argument("--multi_encoder_type", default='single', choices=['single', 'multi', 'split'], type=str, help="Specifies the encoder type to be used for the multi joint condition.")
    group.add_argument("--enc_layers_target", default=1, type=int, help="Num target encoder layers")

    # Prefix completion model
    group.add_argument("--context_len", default=0, type=int, help="If larger than 0, will do prefix completion.")
    group.add_argument("--pred_len", default=0, type=int, help="If context_len larger than 0, will do prefix completion. If pred_len will not be specified - will use the same length as context_len")
    
def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--use_cache", action='store_true', help="If true, will use the cached datasets. Disable to ensure the correct data is loaded.")
    # PRIOR dataset
    group.add_argument(f"--{Pfx.PRIOR}_name", default=Pfx.PRIOR, type=str, choices=Pfx.STREAMS, help="Name of the stream")
    group.add_argument(f"--{Pfx.PRIOR}_dataset", default='humanml', choices=['humanml'], type=str, help="Prior domain dataset")
    group.add_argument(f"--{Pfx.PRIOR}_unconstrained", default=False, type=bool, help="If true, will force unconstrained training on the prior stream.")
    group.add_argument(f"--{Pfx.PRIOR}_checkpoint", default='', type=str, help="If not empty, will load the specified checkpoint and use it as a fixed prior model.")
    # TARGET dataset
    group.add_argument(f"--{Pfx.TARGET}_name", default=Pfx.TARGET, type=str, choices=Pfx.STREAMS, help="Name of the stream")
    group.add_argument(f"--{Pfx.TARGET}_dataset", default='ntu-vibe', choices=['ntu60', 'ntu120', 'ntu-vibe'], type=str, help="Target domain dataset")
    group.add_argument(f"--{Pfx.TARGET}_unconstrained", default=False, type=bool, help="If true, will force unconstrained training on the target stream.")
    group.add_argument(f"--{Pfx.TARGET}_cond_mode", default='mixed', type=str, choices=['action', 'text', 'mixed', 'mixed_token'], help="Target stream conditioning mode. Mixed conditioning uses both action labels and text.")
    group.add_argument(f"--{Pfx.TARGET}_data_rep", default='hml_vec', type=str, choices=["hml_vec", "xyz"], help="Data representation format for target stream")
    group.add_argument(f"--{Pfx.TARGET}_task_split", default='xsub', type=str, help="Specifies the task split to use for the target dataset.")
    group.add_argument(f"--{Pfx.TARGET}_fewshot_id", default='S0000', type=str, help="Specifies the id of the fewshot split to use. (NONE to use the whole task split). ")
    group.add_argument(f"--{Pfx.TARGET}_use_stats", default='HumanML3D', type=str, choices=[None, 'HumanML3D'], help="If a dataset is specified, will use its stats for normalization")
    group.add_argument(f"--{Pfx.TARGET}_checkpoint", default='', type=str, help="If not empty, will load the specified checkpoint and use it as a fixed target model.")

def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str, help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true', help="If set, will overwrite the save_dir if it exists.")
    group.add_argument("--oversample", default=None, type=int, help="Fix the size of the dataset by oversampling to the specified number of samples.")
    group.add_argument("--lr", default=2e-5, type=float, choices=[1e-5, 2e-5, 5e-5, 1e-4], help="Learning rate.")
    group.add_argument("--weight_decay", default=1e-5, type=float, help="Optimizer weight decay.")
    group.add_argument("--grad_clip", default=1.0, type=float, help="Gradient clipping for the generator. negative for no clipping.")
    group.add_argument("--lr_judge_ratio", default=0.2, type=float, help="Fraction of the --lr to be used as judge's Learning rate (if used).")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int, help="Batch size during evaluation loop. Do not change this unless you know what you are doing. T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='val', choices=['val', 'test'], type=str, help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true', help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int, help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=600, type=int, help="Crop size of eval set, If (-1/None) will use all samples.")
    group.add_argument("--log_interval", default=25, type=int, help="Log losses each N steps")
    group.add_argument("--save_interval", default=500, type=int, help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=5_000, type=int, help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int, help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str, help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--starting_checkpoint", default="", type=str, help= "If not empty, will start from the specified pre-training model.")
    group.add_argument("--gen_during_training", action='store_true', help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=12, type=int, help="Number of samples to sample while generating")
    group.add_argument("--gen_num_repetitions", default=5, type=int, help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--gen_guidance_param", default=2.5, type=float, help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--gen_motion_length", default=3.5, type=float, help="The length of the sampled motion [in seconds].")
    group.add_argument("--avg_model_beta", default=0.9, type=float, help="Average model beta (for EMA).")
    group.add_argument("--adam_beta2", default=0.999, type=float, help="Adam beta2.")
    group.add_argument("--joint_names_target", default='DIMP_FINAL', type=str, help="Force single joint configuration by specifing the joints (coma separated). If None - will use the random mode for all end effectors.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'],  help="Sets the source of the init frames, either from the dataset or isaac init poses.")


def add_peft_options(parser): # Parameter Efficient Fine-Tuning options [LoRA, MoE, etc.]
    group = parser.add_argument_group('peft')
    group.add_argument("--peft", nargs='*', type=str, default=[], choices=Pfx.ADAPTERS, help="Type of PEFT to use. (LoRA, MoE, both, ...). ")
    # LoRA options
    group.add_argument(f"--{Pfx.LORA}_name", default=Pfx.LORA, type=str, help="Name of the LoRA adapter. (Has to be the same as the Object name in the model).")
    group.add_argument(f"--{Pfx.LORA}_stream", nargs='*', type=str, default=[Pfx.TARGET], choices=[Pfx.TARGET, Pfx.PRIOR], help="Name of the streams to apply LoRA to.")
    group.add_argument(f"--{Pfx.LORA}_where", nargs='*', type=str, default=['transformer_attn', 'transformer_ff', 'input_process', 'output_process', 'timestep_embed'], choices=['transformer_attn', 'transformer_ff', 'input_process', 'output_process', 'timestep_embed'], help="Where to apply LoRA within MDM.")
    # LoRA Adapter opt.
    group.add_argument(f"--{Pfx.LORA}_rank", default=16, type=int, help="Rank of the LoRA layers.")
    group.add_argument(f"--{Pfx.LORA}_alpha", default=32, type=int, help="Alpha of the LoRA layers.")
    group.add_argument(f"--{Pfx.LORA}_dropout", default=0.1, type=float, help="Dropout of the LoRA layers.")
    group.add_argument(f"--{Pfx.LORA}_bias", default=False, type=bool, help="If true, will train pre-train biases.")
    # MoE options
    group.add_argument(f"--{Pfx.MOE}_name", default=Pfx.MOE, type=str, help="Name of the MoE adapter. (Has to be the same as the Object name in the model).")
    group.add_argument(f"--{Pfx.MOE}_where", nargs='*', type=str, default=[], choices=['transformer_attn', 'transformer_ff'], help="Where to apply MoE within MDM.")
    group.add_argument(f"--{Pfx.MOE}_stream", nargs='*', type=str, default=[Pfx.TARGET, Pfx.PRIOR], choices=[Pfx.TARGET, Pfx.PRIOR], help="Name of the streams to apply MoE to.")
    # MoE Adapter opt.
    group.add_argument(f"--{Pfx.MOE}_num_experts", default=6, type=int, help="Number of experts in the MoE layers.")
    group.add_argument(f"--{Pfx.MOE}_top_k", default=2, type=int, help="Top k experts to use in the MoE layers.")
    group.add_argument(f"--{Pfx.MOE}_lora_rank", default=16, type=int, help="Rank of the LoRA layers within MoE.")
    group.add_argument(f"--{Pfx.MOE}_lora_alpha", default=32, type=int, help="Alpha of the LoRA layers within MoE.")
    group.add_argument(f"--{Pfx.MOE}_lora_dropout", default=0.1, type=float, help="Dropout of the LoRA layers within MoE.")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str, help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str, help="Path to results dir (auto created by the script). If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=9, type=int, help="Maximal number of prompts to sample, if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=5, type=int, help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float, help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], help="Sets the source of the init frames, either from the dataset or isaac init poses.")
    # .
    group.add_argument("--num_rows", default=12, type=int, help="Number of rows in the visualization grid. (in inspect synth dataset)")
    group.add_argument("--num_cols", default=5, type=int, help="Number of columns in the visualization grid. (in inspect synth dataset)")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    # NEW
    group.add_argument("--unconstrained_sampling", action='store_true', help="If true, will sample without any condition.")
    group.add_argument("--sampling_mode", default='single', type=str, choices=['single', 'cycle'], help="Specifies the sampling mode. single - will sample from a single stream, cycle - will sample from both streams in a cycle.")
    group.add_argument("--sampling_stream", default=Pfx.TARGET, type=str, choices=Pfx.STREAMS, help="Specifies which stream to sample from. Ignored if cycle_sampling is set.")
    group.add_argument("--oversample", default=2.0, type=float, help="Factor to increase the number of generated sampled applied to the available train data")
    # OLD
    group.add_argument("--motion_length", default=2.5, type=float, help="The length of the sampled motion [in seconds]. Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str, help="Path to a text file lists text prompts to be synthesized.")
    group.add_argument("--action_file", default='', type=str, help="Path to a text file lists action names to be synthesized.")
    group.add_argument("--text_prompt", default='', type=str, help="A text prompt to be generated.")
    group.add_argument("--action_name", default='', type=str, help="An action name to be generated.")
    group.add_argument("--action_id", nargs='*', type=int, default=[], help="Action IDs to be generated.")
    group.add_argument("--joint_names_target", default='DIMP_FINAL', type=str, help="Force single joint configuration by specifing the joints (coma separated). If None - will use the random mode for all end effectors.")

def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str, help="Path to model####.pt file to be sampled.")
    group.add_argument("--guidance_param", default=2.5, type=float, help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")

def get_cond_mode(stream_args):
    if stream_args.unconstrained:
        cond_mode = 'no_cond'
    elif stream_args.dataset in ['kit', 'humanml']:
        cond_mode = 'text'
    elif stream_args.dataset in ['ntu60', 'ntu120', 'ntu-vibe']:
        # such datasets have selectable conditioning.
        # e.g. in ntu60 we can either have the action label or a natural language translation of the action label.
        cond_mode = stream_args.cond_mode 
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_peft_options(parser)

    args = parser.parse_args()
    args = Pfx.wrap_args_by_prefix(args)
    return apply_rules(args) 

def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)

    args = parse_and_load_from_model(parser)
    args = Pfx.wrap_args_by_prefix(args)
    return apply_rules(args)

def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    args = parse_and_load_from_model(parser)
    args = Pfx.wrap_args_by_prefix(args)
    return args