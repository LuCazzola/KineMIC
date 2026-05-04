import json
from argparse import Namespace
import re
from os.path import join as pjoin
from data_loaders.unified.utils.word_vectorizer import POS_enumerator

def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device, data_stream_args, data_rep, synth_data_folder=None):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = bool(value == 'True')
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)
    # print(opt)
    opt_dict['which_epoch'] = 'latest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.task_split = getattr(data_stream_args, 'task_split', None)
    opt.fewshot_id = getattr(data_stream_args, 'fewshot_id', None)

    # Few-shot options, they can be omitted
    base_dataset_root = './dataset'
    if opt.dataset_name == 't2m':
        # HumanML3D dataset
        opt.data_root = pjoin(base_dataset_root, 'HumanML3D')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.mean, opt.std = 'Mean', 'Std'
        opt.mean_eval, opt.std_eval = 't2m_mean', 't2m_std'
        opt.joints_num = 22
        opt.dim_pose, opt.dim_pose_eval = 263, 263
        opt.max_motion_length = 196
    elif opt.dataset_name in ['ntu60', 'ntu120', 'ntu-vibe']:
        # NTU RGB+D dataset
        opt.data_root = pjoin(base_dataset_root, opt.dataset_name.upper())
        opt.metadata = pjoin(base_dataset_root, 'ntu_metadata.json') # maps class names to captions
        opt.classes = pjoin(base_dataset_root, 'ntu_classes.txt')
        opt.joints_num = 22
        opt.max_motion_length = 196

        if data_rep == 'hml_vec':
            opt.motion_format_dir = 'new_joint_vecs'
            opt.mean, opt.std = 'Mean_joint_vecs', 'Std_joint_vecs'
            opt.dim_pose = 263
        elif data_rep == 'xyz':
            opt.motion_format_dir = 'new_joints'
            opt.mean, opt.std = 'Mean_joints', 'Std_joints'
            opt.dim_pose = 66
        opt.dim_pose_eval = 66 # Evaluators always use xyz format

        # Folder pointing to the Motion samples .npy files
        if synth_data_folder is None:
            opt.motion_dir = pjoin(opt.data_root, opt.motion_format_dir)
        else:
            # NOTE: synth assumes FewShot mode
            opt.motion_dir = pjoin(opt.data_root, 'splits', 'synth', opt.fewshot_id, opt.task_split, synth_data_folder, 'new_joint_vecs')

        opt.mean_eval, opt.std_eval = f'{opt.dataset_name}_{opt.task_split}_mean', f'{opt.dataset_name}_{opt.task_split}_std'
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.default_root = pjoin(opt.data_root, 'splits', 'default')
        opt.fewshot_root = pjoin(opt.data_root, 'splits', 'fewshot')
        opt.synth_root = pjoin(opt.data_root, 'splits', 'synth')
        opt.meta_file = 'meta.json' # generation details of the few-shot (if in few-shot mode)        
        opt.info_file = 'info.json' # dataset info file (for synth data)
        
        if data_stream_args.fewshot_id is None:
            opt.data_root = pjoin(opt.default_root, opt.task_split) # Full data
        else:
            if synth_data_folder is None:
                opt.data_root = pjoin(opt.fewshot_root, opt.fewshot_id, opt.task_split)
            else:
                opt.data_root = pjoin(opt.synth_root, opt.fewshot_id, opt.task_split, synth_data_folder)
    else:
        raise KeyError(f'Dataset "{opt.dataset_name}" not recognized')

    # Stats loading (Mean and Std) Allowing use of pretrained stats
    use_stats = getattr(data_stream_args, 'use_stats', None)
    if use_stats is None:
        opt.stats_from_pretrain = False
        if opt.dataset_name in ['t2m', 'kit']:
            opt.stats_root = opt.data_root
        elif opt.dataset_name in ['ntu60', 'ntu120', 'ntu-vibe']:
            opt.stats_root = pjoin(opt.default_root, opt.task_split)
        else:
            raise KeyError(f'Dataset "{opt.dataset_name}" does not support pretraining')
    else:
        opt.stats_from_pretrain = True
        if use_stats == 'HumanML3D':
            opt.pretrain_dataset_name = 't2m'
            opt.stats_root = pjoin(base_dataset_root, 'HumanML3D')
            opt.mean, opt.std = 'Mean', 'Std'
        else:
            raise KeyError(f'Pretrain dataset "{use_stats}" not recognized')
    
    # Retrieve action captions magging file if provided
    metadata = getattr(opt, 'metadata', None)
    if metadata :
        opt.caption_2_action, opt.action_id_2_action_name = {}, {}
        with open(metadata, 'r') as f:
            mapping = json.load(f)
        for action_id, entry in enumerate(mapping['actions']):
            for caption in entry['captions']:
                opt.caption_2_action[caption.lower()] = action_id
            opt.action_id_2_action_name[action_id] = entry['action']
    else:
        opt.caption_2_action = None

    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt