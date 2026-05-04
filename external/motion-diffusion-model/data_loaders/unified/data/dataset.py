from argparse import Action
from locale import normalize
from re import A
import torch
import json
from torch.utils.data import Dataset
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from utils import dist_util

import clip
from sklearn.neighbors import NearestNeighbors

from data_loaders.unified.utils.word_vectorizer import WordVectorizer
from data_loaders.unified.utils.get_opt import get_opt
from .aug import MotionDataAugmenter

from typing import Union

class MotionDataset(Dataset):
    """
    Base class for all skeleton-based conditioned motion synthesis datasets.
    """
    def __init__(self, opt, mean, std, split_file, no_motion=False):
        # Init. vars
        self.max_motion_length = opt.max_motion_length
        self.min_motion_len = 40 if opt.dataset_name =='t2m' else 0 # only to HumanML3D we apply a min length filter
        self.max_length = opt.fixed_len if opt.fixed_len > 0 else 10
        self.pointer = 0
        self.fixed_length = 120 if no_motion else None
        self.opt = opt
        self.split_file = split_file

        # Gather filenames of motion data
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        data_dict = {}
        length_list, new_name_list, skipped_samples = [], [], []
        if not no_motion: # Load motion data
            def load_motion(samples, samples_dir, desc='Motion'):
                # Load from scratch
                for name in tqdm(samples, desc=desc):
                    if name in opt.blacklist:
                        skipped_samples.append(name)
                        continue # samples to ignore
                    
                    motion = np.load(pjoin(samples_dir, name + '.npy'))
                    motion_length = len(motion)
                    if not (self.min_motion_len <= motion_length < 200):
                        skipped_samples.append(name)
                        continue

                    data_dict[name] = {
                        'motion': motion,
                        'length': motion_length,
                    }
                    length_list.append(motion_length)
                    new_name_list.append(name)
            
            # Only real motion data
            load_motion(id_list, opt.motion_dir, desc='Motions')
            print("> Samples : ", len(new_name_list))
            if opt.synth_data_folder is not None:
                print("> Using Syntehtic Data !!!")
            
            # sort by motion length
            new_name_list, length_list = map(list, zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1])))
        else:
            # No motion data
            for name in tqdm(id_list, desc='No-Motion'):
                if name in opt.blacklist:
                    skipped_samples.append(name)
                    continue # samples to ignore
                new_name_list.append(name)
                length_list.append(self.fixed_length)
                data_dict[name] = {'motion': None, 'length': None}
        print(f'Skipped [{len(skipped_samples)}] samples in total')
        
        # Default data transformations
        self.transforms = MotionDataAugmenter(opt)
        self.transforms.set_transforms([
            self.transforms.to_unit_length, # -> makes the motion multiples of defined unit size
            lambda m, m_len: self.normalize_motion(m) # -> Z-score normalization
        ])

        # Store core components
        self.data_dict = data_dict
        self.mean, self.std = mean, std
        self.name_list = new_name_list
        self.length_list = length_list
        self.no_motion = no_motion
        # other vars  (for subclasses)
        self.full_name_list = id_list # all names (including those that might have been discarded)

    def __len__(self):
        return len(self.name_list) - self.pointer

    def _get_data(self, item):
        """Get data item by index."""
        idx = self.pointer + item
        key = self.name_list[idx]
        return self.data_dict[key]

    def de_normalize_motion(self, x):
        """Z-de-normalization of motion data."""
        return x * self.std + self.mean
    
    def normalize_motion(self, x):
        """Z-normalization of motion data."""
        return (x - self.mean) / self.std

    def transform_motion(self, motion, m_length):
        """motion pre-processing"""

        if self.no_motion: # for no-motion datasets
            return None, None
        
        # Apply specified transformations
        original_length = m_length
        motion = self.transforms(motion, m_length)
        m_length = len(motion)
        
        # Pad to max_motion_length
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                    np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                    ], axis=0)
        length = (original_length, m_length) if self.opt.fixed_len > 0 else m_length

        return motion, length
        
    def __getitem__(self, item) :
        data = self._get_data(item)
        motion, m_length = data['motion'], data['length']
        # Apply transformations to motion data
        motion, length = self.transform_motion(motion, m_length)
        return motion, length
    
    def oversample(self, to_length, truncate_to=-1):
        """
        Oversamples the dataset to a target length by duplicating samples.
        This method correctly updates all parallel lists (name_list, length_list).
        """
        # 1. Bundle the parallel data into a list of tuples
        all_samples = list(zip(self.name_list, self.length_list))
        expansion_pool = all_samples
        
        if not expansion_pool:
            print("Warning: Expansion pool is empty. Oversampling cannot be performed.")
            return

        # 2. Repeat the expansion pool and add it to the original samples
        num_repeats = (to_length // len(expansion_pool)) + 1
        expanded_samples = all_samples + expansion_pool * num_repeats
        # 3. Truncate to the desired final length (using the correct variable)
        final_length = truncate_to if truncate_to > 0 else to_length
        final_samples = expanded_samples[:final_length]
        # 4. Unzip the data back into the parallel instance lists
        if final_samples:
            names, lengths = zip(*final_samples)
            self.name_list = list(names)
            self.length_list = list(lengths)
        else:
            self.name_list, self.length_list = [], []        
        

class Action2MotionDataset(MotionDataset):
    """
    Action-Conditioned Skeletal-based Motion Dataset
    """
    def __init__(self, opt, mean, std, split_file, no_motion=False):
        
        super().__init__(opt, mean, std, split_file, no_motion=no_motion)

        classnames = [] # maps raw action id into textual description for the class
        with cs.open(opt.classes, 'r') as f:
            for line in f.readlines():
                classnames.append(line.strip())

        action_id = []
        with cs.open(split_file.replace('.txt', '_y.txt'), 'r') as f:
            for line in f.readlines():
                action_id.append(int(line.strip()))
        
        # gather all unique actions available in the dataset
        self.all_actions = sorted(set(a_id for idx, a_id in enumerate(action_id) if self.full_name_list[idx] in self.data_dict))
        print(f'> Found [{len(self.all_actions)}] unique actions ...')
        print(f'> Available actions:')
        for a_id in set(self.all_actions):
            print(f'>> {a_id+1} -> {classnames[a_id]}')
        print("")
    
        # Build mapping dicts
        self._compact_id_map = {a_id: idx for idx, a_id in enumerate(self.all_actions)} # OG id -> compact id
        self._compact_id_map_rev = {idx: a_id for idx, a_id in enumerate(self.all_actions)} # compact id -> OG id
        self._classname = {a_id: classnames[a_id] for a_id in self.all_actions}
        self._classname_rev = {a_name: a_id for a_id, a_name in self._classname.items()} # reverse mapping

        # store to data_dict
        for idx, name in enumerate(self.full_name_list):
            if name in self.data_dict.keys():
                action = action_id[idx]
                self.data_dict[name]['action'] = self._compact_id_map[action]
                self.data_dict[name]['action_text'] = self._classname[action]

        # store core components
        self.num_actions = len(self.all_actions)

    def __getitem__(self, item):
        # Gather Motion-Data
        motion, m_length = super().__getitem__(item)
        # Gather Action-Data
        data = self._get_data(item)
        action, action_text = data['action'], data['action_text']
        return motion, m_length, action, action_text, self.name_list[self.pointer + item] # sample id as debug key

    def get_compact_class_id(self, id, reverse=False):
        if reverse:
            return self._compact_id_map_rev[id] # Compact -> Original
        return self._compact_id_map[id] # Original -> Compact
    
    def get_class_name(self, id, is_compact_id=False, reversed=False):
        if reversed:
            assert isinstance(id, str)
            return self._classname_rev[id] # class_name -> id
        assert isinstance(id, int)
        class_id = id if not is_compact_id else self.get_compact_class_id(id, reverse=True)
        return self._classname[class_id] # id -> class name


class Text2MotionDataset(MotionDataset):
    """
    Text-Conditioned Motion Dataset
    """
    def __init__(self, opt, mean, std, split_file, no_motion=False):
        
        super().__init__(opt, mean, std, split_file, no_motion=no_motion)

        partial_sequences_created = 0
        new_entries = {} 
        entries_to_remove = set()
        all_used_names = set(self.data_dict.keys())
        
        # Process each motion file
        for name, data in tqdm(self.data_dict.items(), desc='Texts'):
            full_sequence_texts = []  # Texts for the full motion (f_tag=0, to_tag=0)
            has_full_sequences, has_partial_sequences = False, False
            # Read text file for this motion
            with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                for line in f.readlines():
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    text_dict = {'caption': caption, 'tokens': tokens}

                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag
                    
                    if f_tag == 0.0 and to_tag == 0.0:
                        has_full_sequences = True
                        full_sequence_texts.append(text_dict)
                    else:
                        has_partial_sequences = True
                        start_frame = int(f_tag * 20)
                        end_frame = int(to_tag * 20)
                        partial_motion = data['motion'][start_frame:end_frame]
                        partial_length = len(partial_motion)
                        
                        if not (self.min_motion_len <= partial_length < 200):
                            continue
                        
                        vocab = 'ABCDEFGHIJKLMNOPQRSTUVW'
                        for idx, c in enumerate(vocab):
                            new_name = f'{c}_{name}'
                            if new_name not in all_used_names:
                                break
                            assert idx < len(vocab)-1, 'Too many partial sequences for one motion, cannot assign new name'

                        all_used_names.add(new_name)                        
                        new_entries[new_name] = {
                            'motion': partial_motion,
                            'length': partial_length,
                            'text': [text_dict]
                        }
                        partial_sequences_created += 1
            
            if has_full_sequences:
                # Store on 'text' field the captions relative
                # to the full sequence
                self.data_dict[name]['text'] = full_sequence_texts
            
            elif has_partial_sequences :
                # If a motion had only partial sequences (no f_tag == 0.0 and to_tag == 0.0)
                # we should remove the entry, as it's unused
                entries_to_remove.add(name)

        if partial_sequences_created > 0:
            print(f'> Created [{partial_sequences_created}] partial sequences')
            for name in entries_to_remove:
                # clean up entries
                del self.data_dict[name]
                dead_entry_idx = self.name_list.index(name)
                self.name_list.pop(dead_entry_idx)
                self.length_list.pop(dead_entry_idx)
            # update the data_dict with new entries
            self.data_dict.update(new_entries)
            for name, data in new_entries.items(): # update name and length lists
                self.name_list.append(name)
                self.length_list.append(data['length'])
            # sort by length
            self.name_list, self.length_list = map(list, zip(*sorted(zip(self.name_list, self.length_list), key=lambda x: x[1])))
            
        # Store core components
        self.w_vectorizer = WordVectorizer(pjoin(opt.cache_dir, 'glove'), 'our_vab')


    def transform_text(self, text_list):
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.no_motion:  # for no-motion datasets
            return None, None, caption, self.fixed_length, None # NOTE: fixed_length should be set from outside before sampling

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]    
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
            
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, tokens

    def __getitem__(self, item):

        motion, m_length = super().__getitem__(item)
        data = self._get_data(item)

        word_embeddings, pos_one_hots, caption, sent_len, tokens = self.transform_text(data['text'])
        tokens = '_'.join(tokens) if tokens else None

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens

class MixedTextAction2MotionDataset(MotionDataset):
    def __init__(self, opt, mean, std, split_file, no_motion=False):
        
        super().__init__(opt, mean, std, split_file, no_motion=no_motion)
        text_dataset = Text2MotionDataset(opt, mean, std, split_file, no_motion=True) # Load conditioning-only text dataset
        action_dataset = Action2MotionDataset(opt, mean, std, split_file, no_motion=True) # Load conditioning-only action dataset
        
        assert set(self.name_list) == set(text_dataset.name_list) == set(action_dataset.name_list), 'Name lists should have coherent content'
    
        # update data_dict with text and action info
        for name in self.name_list:
            self.data_dict[name]['text'] = text_dataset.data_dict[name]['text']
            self.data_dict[name]['action'] = action_dataset.data_dict[name]['action']
            self.data_dict[name]['action_text'] = action_dataset.data_dict[name]['action_text']
    
        # special token tagging [A001], [A002], ... it's 1 based
        self.tag = [f'[A{(a+1):03d}]' for a in action_dataset.all_actions] 

        # action-related attr.
        for attr in ['num_actions', 'all_actions',
                    '_compact_id_map', '_compact_id_map_rev', '_classname', '_classname_rev',
                    'get_compact_class_id', 'get_class_name']:
            setattr(self, attr, getattr(action_dataset, attr))
        
        # text-related attr.
        for attr in ['w_vectorizer', 'transform_text']:
            setattr(self, attr, getattr(text_dataset, attr))

    def __getitem__(self, item):
        # Gather Motion-Data
        motion, m_length = super().__getitem__(item)
        # Gather Action-Data
        data = self._get_data(item)
        action, action_text = data['action'], data['action_text']
        # Gather Text-Data
        word_embeddings, pos_one_hots, caption, sent_len, tokens = self.transform_text(data['text']) # type: ignore
        tokens = '_'.join(tokens) if tokens else None
        # return everything
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens, action, action_text, self.tag[action]

class DualMDMDataset(Dataset):
    def __init__(self, prior_dataset: Text2MotionDataset, target_dataset: Union[Text2MotionDataset, Action2MotionDataset, MixedTextAction2MotionDataset], top_k_sp: int):
        super().__init__()

        self.prior_dataset = prior_dataset
        self.target_dataset = target_dataset
        self.k = top_k_sp

        # Initialize the target text dataset for processing
        if isinstance(target_dataset, Action2MotionDataset):
            self.target_text_dataset = Text2MotionDataset(target_dataset.opt, target_dataset.mean, target_dataset.std, target_dataset.split_file, no_motion=True)
        else:
            self.target_text_dataset = target_dataset

        # Load the CLIP model
        self.clip_model, _ = clip.load("ViT-B/32", device=dist_util.dev())
        self.clip_model.eval()
        # Process prior dataset captions and get averaged embeddings
        prior_embeds = self._get_averaged_prior_embeddings()
        # Build nearest neighbor index on the prior embeddings
        self.nn_index = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        self.nn_index.fit(prior_embeds)
        # Process target dataset captions and build top-k matchings map
        self.top_k_matchings = self._get_target_matchings()
        # Release the CLIP model from memory
        del self.clip_model

        # for consistency
        self.mean_gpu = torch.tensor(self.target_dataset.mean).to(dist_util.dev())[None, :, None, None]
        self.std_gpu = torch.tensor(self.target_dataset.std).to(dist_util.dev())[None, :, None, None]

        
    def _get_averaged_prior_embeddings(self, batch_size=64):
        """
        Gathers all captions from the prior dataset, encodes them in batches
        using CLIP, and averages the embeddings for each motion.
        """
        print("> Computing CLIP embeddings for [prior] texts ...")
        all_prior_captions = []
        prior_motion_to_caption_map = {}
        current_idx = 0

        # Gather all captions and map motion names to their caption indices
        for name in self.prior_dataset.name_list:
            entry = self.prior_dataset.data_dict[name]
            captions = [text['caption'] for text in entry['text']]
            all_prior_captions.extend(captions)
            prior_motion_to_caption_map[name] = (current_idx, current_idx + len(captions))
            current_idx += len(captions)

        # Encode all captions in batches
        all_prior_embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_prior_captions), batch_size):
                batch = all_prior_captions[i:i + batch_size]
                tokens = clip.tokenize(batch, truncate=True).to(dist_util.dev())
                embeddings = self.clip_model.encode_text(tokens)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_prior_embeddings.append(embeddings.cpu())
        all_prior_embeddings = torch.cat(all_prior_embeddings, dim=0)
        
        # Average embeddings for each motion
        prior_embeds = []
        for name in self.prior_dataset.name_list:
            start_idx, end_idx = prior_motion_to_caption_map[name]
            motion_embeddings = all_prior_embeddings[start_idx:end_idx]
            mean_embedding = motion_embeddings.mean(dim=0, keepdim=True)
            prior_embeds.append(mean_embedding)
        return torch.cat(prior_embeds, dim=0)

    def _get_target_matchings(self):
        """
        For each target sample's caption, finds and stores its top-k matches
        using the pre-built nearest neighbor index.
        """
        print("> Building a top-k map for each [target] sample's caption ...")
        top_k_matchings = {}
        with torch.no_grad():
            for target_name in self.target_text_dataset.name_list:
                entry = self.target_text_dataset.data_dict[target_name]
                captions = [text['caption'] for text in entry['text']]
                
                caption_matches = []
                for caption in captions:
                    tokens = clip.tokenize([caption], truncate=True).to(dist_util.dev())
                    embedding = self.clip_model.encode_text(tokens)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    # Find top-k nearest neighbors
                    _, indices = self.nn_index.kneighbors(embedding.cpu(), return_distance=True)
                    caption_matches.append(indices.flatten())
                
                top_k_matchings[target_name] = caption_matches
        return top_k_matchings

    def __len__(self):
        return len(self.target_dataset)
    
    def __getitem__(self, item):
        '''
        Returns a tuple of (prior_data, target_data)
        > The selection of the target_data is stratified
        1. sample randomly a top_k list (there's one per caption associate to the sample)
        2. from the list randomly sample one index and retrieve the corresponding prior data
        '''
        target_data = self.target_dataset.__getitem__(item)
        target_name = self.target_dataset.name_list[item]
        # Randomly choose one of the top-k lists associated with the target's captions
        chosen_candidate_list = random.choice(self.top_k_matchings[target_name])
        # Get an item from the sampled list and retrieve the corresponding prior data
        p_item = random.choice(chosen_candidate_list)
        prior_data = self.prior_dataset.__getitem__(p_item)
        return prior_data, target_data

    def normalize_motion(self, x, gpu=False):
        if gpu: return (x - self.mean_gpu) / self.std_gpu
        return self.target_dataset.normalize_motion(x)

    def de_normalize_motion(self, x, gpu=False):
        if gpu: return x * self.std_gpu + self.mean_gpu
        return self.target_dataset.de_normalize_motion(x)

class MotionDatasetWrapper(Dataset):
    def __init__(self,
        mode,
        datapath='',
        split='',
        cond_mode='',
        data_rep='',
        data_stream_args=None,
        **kwargs
    ):
        assert split in ['train', 'val', 'test'], 'Unknown split: {}'.format(split)
        assert mode in ['train', 'eval', 'gt', 'cond_only'], 'Unknown mode: {}'.format(mode)
        assert cond_mode in ['raw', 'text', 'action', 'mixed'], 'Unknown cond_mode: {}'.format(cond_mode)
        self.mode = mode
        self.cond_mode = cond_mode
        self.data_rep = data_rep
        print(f"> Data rep. [{self.data_rep}]")
        print(f"> Conditioning [{cond_mode}]")

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = kwargs.get('abs_path', '.')
        synth_data_folder = kwargs.get('synth_data_folder', None)
        device = kwargs.get('device', None)
        opt = get_opt(pjoin(abs_base_path, datapath), device, data_stream_args, data_rep, synth_data_folder=synth_data_folder) # Parse option .txt file
        
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.meta_dir = pjoin(abs_base_path, './dataset')
        opt.data_root = pjoin(abs_base_path, opt.data_root)        
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.cache_dir = kwargs.get('cache_path', '.')
        opt.fixed_len = kwargs.get('fixed_len', 0)
        opt.blacklist = kwargs.get('blacklist', set()) # samples to ignore within the parsed dataset
        opt.synth_data_folder = synth_data_folder
        if opt.fixed_len > 0:
            opt.max_motion_length = opt.fixed_len
        
        print("> Task: ", getattr(data_stream_args, 'task_split', 'base'))
        print("> Subset: ", getattr(data_stream_args, 'fewshot_id', 'Full'))
        print(f'[Dataset: {opt.dataset_name} | Mode: {mode} | Split: {split}]')
        # Set normalization stats
        if mode == 'gt':
            # used by T2M models (including evaluators) (cached GT)
            if getattr(opt, 'has_eval_norm', True) :
                print(f'GT Stats for Evaluator from {opt.meta_dir} [{opt.mean_eval}, {opt.std_eval}]')
                self.mean = np.load(pjoin(opt.meta_dir, opt.mean_eval + '.npy'))
                self.std = np.load(pjoin(opt.meta_dir, opt.std_eval + '.npy'))
            else:
                print(f'GT Stats for Evaluator : NO Normalization (Mean=0.0, Std=1.0)]')
                self.mean = np.zeros((opt.dim_pose_eval,))
                self.std = np.ones((opt.dim_pose_eval,))

        elif mode in ['train', 'eval', 'cond_only']:
            # Load dataset stats from the data root
            stats_root = pjoin('.', opt.stats_root)  
            print(f'Stats from {stats_root} [{opt.mean}, {opt.std}]')
            self.mean = np.load(pjoin(stats_root, opt.mean + '.npy'))
            self.std = np.load(pjoin(stats_root, opt.std + '.npy'))
            
        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            if getattr(opt, 'has_eval_norm', True) :
                print(f'Stats for Evaluator from {opt.meta_dir} [{opt.mean_eval}, {opt.std_eval}]')
                self.mean_for_eval = np.load(pjoin(opt.meta_dir, opt.mean_eval + '.npy'))
                self.std_for_eval = np.load(pjoin(opt.meta_dir, opt.std_eval + '.npy'))
            else:
                print(f'Stats for Evaluator : NO Normalization (Mean=0.0, Std=1.0)]')
                self.mean_for_eval = np.zeros((opt.dim_pose_eval,))
                self.std_for_eval = np.ones((opt.dim_pose_eval,))

        # needed for some losses computations, tailoerd to MDM output format
        self.mean_gpu = torch.tensor(self.mean).to(device)[None, :, None, None]
        self.std_gpu = torch.tensor(self.std).to(device)[None, :, None, None]

        # Caching
        self.split_file = pjoin(
            opt.data_root,
            f'{os.path.basename(split)}.txt' if synth_data_folder is None else 'synth.txt'
        )

        no_motion = (mode == 'cond_only') # for conditioning-only datasets
        if cond_mode == 'raw':
            self.m_dataset = MotionDataset(opt, self.mean, self.std, self.split_file, no_motion=no_motion)
        elif cond_mode == 'text':
            self.m_dataset = Text2MotionDataset(opt, self.mean, self.std, self.split_file, no_motion=no_motion)
            self.w_vectorizer = self.m_dataset.w_vectorizer # FIXME not sure this is necessary (for compatibility)
        elif cond_mode == 'action':
            self.m_dataset = Action2MotionDataset(opt, self.mean, self.std, self.split_file, no_motion=no_motion)
        elif cond_mode == 'mixed':
            self.m_dataset = MixedTextAction2MotionDataset(opt, self.mean, self.std, self.split_file, no_motion=no_motion)
            self.w_vectorizer = self.m_dataset.w_vectorizer
        else :
            raise ValueError('Unknown cond_mode: {}'.format(cond_mode))

        self.num_actions = getattr(self.m_dataset, 'num_actions', 1)
        self.opt = opt

        assert len(self.m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'
        
        print("> Dataset size: ", len(self.m_dataset))


    def set_transforms(self, transforms):
        self.m_dataset.transforms.set_transforms(transforms)

    def de_normalize_motion(self, x, gpu=False):
        if gpu: return x * self.std_gpu + self.mean_gpu
        return self.m_dataset.de_normalize_motion(x)

    def normalize_motion(self, x, gpu=False):
        if gpu: return (x - self.mean_gpu) / self.std_gpu
        return self.m_dataset.normalize_motion(x)

    def __getitem__(self, item):
        return self.m_dataset.__getitem__(item)

    def __len__(self):
        return self.m_dataset.__len__()

class HumanML3D(MotionDatasetWrapper):
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split='train', cond_mode='text', data_rep='hml_vec', **kwargs):
        super().__init__(mode, datapath, split, cond_mode, data_rep, **kwargs)

class KIT(MotionDatasetWrapper):
    def __init__(self, mode, datapath='./dataset/kit_opt.txt', split='train', cond_mode='text', data_rep='hml_vec', **kwargs):
        super().__init__(mode, datapath, split, cond_mode, data_rep, **kwargs)

class NTU60(MotionDatasetWrapper):
    def __init__(self, mode, datapath='./dataset/ntu60_opt.txt', split='train', cond_mode='action', data_rep='hml_vec', **kwargs):
        super().__init__(mode, datapath, split, cond_mode, data_rep, **kwargs)

class NTU120(MotionDatasetWrapper):
    def __init__(self, mode, datapath='./dataset/ntu120_opt.txt', split='train', cond_mode='action', data_rep='hml_vec', **kwargs):
        super().__init__(mode, datapath, split, cond_mode, data_rep, **kwargs)

class NTUVIBE(MotionDatasetWrapper):
    def __init__(self, mode, datapath='./dataset/ntu-vibe_opt.txt', split='train', cond_mode='action', data_rep='hml_vec', **kwargs):
        super().__init__(mode, datapath, split, cond_mode, data_rep, **kwargs)