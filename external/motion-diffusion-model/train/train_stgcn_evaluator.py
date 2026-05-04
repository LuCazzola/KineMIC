'''
This script is used to train a ST-GCN evaluator on the NTU60 dataset.
* If a fewshot_id is provided, it will train only the classification head on the few-shot classes.
* If no fewshot_id is provided, it will train on the full dataset. (of the given task split, e.g. xsub or xview)
'''

import random
import torch
import os
import json
from os.path import join as pjoin
from argparse import Namespace

from data_loaders.get_data import get_single_stream_dataloader
from utils import dist_util
from train.train_platforms import WandBPlatform, NoPlatform
from utils.parser_util import Pfx
from argparse import ArgumentParser

from utils.dataset_util.ntu_util import get_ntu_blacklist 


from model.motion_encoders import STGCN
from model.wrappers import Classifier
from eval.evaluators.a2m.trainer import TrainerA2M


if __name__ == "__main__":
    
    params = ArgumentParser()
    params.add_argument('--dataset', type=str, default='NTU60', choices=['NTU60', 'NTU120', 'NTU-VIBE'], help='Dataset to use (e.g., ntu60).')
    params.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    params.add_argument('--num_epochs', type=int, default=80, help='Number of epochs to train.')
    params.add_argument('--evaluator', type=str, default='stgcn', choices=['stgcn'],  help='Type of evaluator to train (e.g., stgcn).')
    params.add_argument('--task_split', type=str, default='xsub', choices=['xsub', 'xview'], help='Task split for the dataset (e.g., xsub, xview).')
    params.add_argument('--fewshot_id', type=str, default=None, help='Few-shot ID for few-shot learning scenarios.')
    params.add_argument('--overwrite', action='store_true', help='Whether to overwrite the save directory if it exists.')
    params.add_argument('--train_platform_type', type=str, default='NoPlatform', choices=['NoPlatform', 'WandBPlatform'], help='Type of training platform to use (e.g., WandBPlatform, NoPlatform).')
    params.add_argument('--use_cache', action='store_true', help='Whether to use cached datasets if available.')
    params.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    args = params.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = dist_util.dev()
    stream_args = Namespace(
        name='target',
        dataset=args.dataset.lower(),
        unconstrained=False,
        cond_mode='action',
        data_rep='xyz',
        task_split=args.task_split,
        fewshot_id=None, # Consider the whole dataset
        use_stats='HumanML3D' # this is just a placeholder, we actually don't apply Z-score normalization for the evaluator
    )
    args.stream_args = stream_args
    args.fewshot_id = 'full' if args.fewshot_id is None else args.fewshot_id
    SAVE_DIR = pjoin('.', 'eval', 'evaluators', 'a2m', 'weights', stream_args.dataset, stream_args.task_split, f'{args.evaluator}-{args.fewshot_id}')

    if SAVE_DIR is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(SAVE_DIR) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists. Specify --overwrite to override the save'.format(SAVE_DIR))
    elif not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    train_platform = eval(args.train_platform_type)(SAVE_DIR, f'{args.evaluator}-{stream_args.dataset}-evaluator')
    train_platform.report_args(Pfx.serialize_args(args), name='Args')

    ##
    ## GATHER DATA
    ##

    # A blacklist of samples to exclude
    # it essentially look for all the samples in the task split and returns
    # those that DO NOT belong to the few-shot classes
    blacklist, whitelist = get_ntu_blacklist(args.dataset, args.fewshot_id, args.task_split)

    # Data loaders
    data = {}
    for split, hml_mode in [('train', 'train'), ('val', 'eval')]:
        print("\ncreating [{}] data loader...".format(split))
        data[split] = get_single_stream_dataloader(
            data_stream_args=stream_args, 
            batch_size=args.batch_size, 
            split=split,
            hml_mode=hml_mode,
            device=device,
            evaluator=args.evaluator,
            blacklist=blacklist[split]
        ) # Exclude blacklisted samples
        print("> Num samples: ", len(data[split].dataset))

    # save some metadata
    with open(pjoin(SAVE_DIR, 'info.json'), 'w') as f:
        json.dump({
            'task_split': stream_args.task_split,
            'data_rep': stream_args.data_rep,
            'num_classes': data['train'].dataset.num_actions,
            'class_list': data['train'].dataset.m_dataset.all_actions,
        }, f, indent=4)

    ##
    ## CREATE MODEL AND TRAIN
    ##

    backbone = STGCN(dict(layout='humanml', mode='stgcn_spatial'))
    margs = {'model': backbone, 'in_dim': backbone.out_channels}
    model = Classifier(
        **margs,
        num_classes=data['train'].dataset.num_actions
    ).to(device)
    print(f"\nModel Size: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)} M\n")

    trainer = TrainerA2M(model, data, args.num_epochs, train_platform=train_platform, save_dir=SAVE_DIR)
    trainer.train()
    