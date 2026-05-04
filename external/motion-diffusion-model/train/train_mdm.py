# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args, Pfx
from utils import dist_util
from diffusion import logger
from train.training_loop import TrainLoop
from data_loaders.get_data import get_single_stream_dataloader, get_dual_stream_dataloader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import WandBPlatform, NoPlatform  # required for the eval operation

def main():
    args = train_args()

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists. Change --save_dir or specify --overwrite to override the save'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    fixseed(args.seed)
    logger.configure(dir=args.save_dir)

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir, args.model_type)
    train_platform.report_args(Pfx.serialize_args(args), name='Args')
    
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(Pfx.serialize_args(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)
    print("> Model type: {}".format(args.model_type))
    active_streams = args.stream_names if args.model_type in ['KineMIC'] else [args.single_stream]
    eval_stream = args.single_stream if args.model_type in ['MDM'] else 'target'

    data = {}
    for stream in active_streams :
        print(f"\n=== {stream.upper()} Data ===")
        print(f"creating [{stream}] data loader...")
        stream_args = getattr(args, stream)
        data[stream] = get_single_stream_dataloader( 
            data_stream_args=stream_args,
            batch_size=args.batch_size,
            split='train', hml_mode='train',
            device=dist_util.dev(),
            oversample=args.oversample
        )
    
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data, active_streams)
    
    if args.model_type in ['MDM']:
        data = data[args.single_stream]
    elif args.model_type == 'KineMIC':
        # Unifies the 2 separate streams within the same Dataloader
        print("\n=== KineMIC Dataset ===")
        data = get_dual_stream_dataloader(data['prior'], data['target'], args.top_k_sp)
    else:
        pass

    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('[Total] params: %.3fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))

    print("\nTraining...")
    trainer = TrainLoop(args, train_platform, model, diffusion, data, eval_stream)    
    trainer.run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
