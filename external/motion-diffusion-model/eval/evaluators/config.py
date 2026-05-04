from dataclasses import dataclass, field
from os.path import join as pjoin
from abc import ABC
from data_loaders.unified.utils.word_vectorizer import POS_enumerator
import json
from typing import Optional, List

@dataclass
class BaseEvaluatorConfig(ABC):
    """Base configuration for evaluator parameters"""
    dataset_name: str
    device: str
    checkpoints_dir: str = '.'
    data_dir: str = './dataset'
    task_split: str = ''
    fewshot_id: Optional[str] = None  # None means full dataset

@dataclass
class A2MEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Action-to-Motion evaluator"""
    
    evaluator_type : str = 'stgcn'
    class_list: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Customize config based on dataset"""

        self.checkpoints_dir = pjoin('.', 'eval', 'evaluators', 'a2m', 'weights')

        if self.dataset_name in ['ntu60', 'ntu120', 'ntu-vibe']:

            assert self.task_split in ['xsub', 'xview'], 'Invalid task NTU dataset'
            self.checkpoints_dir = pjoin(self.checkpoints_dir, self.dataset_name, self.task_split)

            if self.fewshot_id is None:
                # all single skeleton classes
                raise NotImplementedError('Full class set not implemented yet for A2M')
            else:
                with open(pjoin(self.data_dir, self.dataset_name.upper(), 'splits', 'fewshot', self.fewshot_id, 'meta.json'), 'r') as f:
                    self.class_list = json.load(f)['class_list'] # pick few-shot classes
        else :
            raise ValueError(f'Unknown dataset name for A2M: {self.dataset_name}')
        
        self.num_classes = len(self.class_list)
            
            
@dataclass
class T2MEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Text-to-Motion evaluator"""
    dim_word: int = 300
    max_motion_length: int = 196
    dim_motion_hidden: int = 1024
    max_text_len: int = 20
    dim_text_hidden: int = 512
    dim_coemb_hidden: int = 512
    dim_movement_enc_hidden: int = 512
    dim_movement_latent: int = 512
    unit_length: int = 4
    
    def __post_init__(self):
        self.dim_pos_ohot = len(POS_enumerator)
        
        if self.dataset_name == 'humanml':
            self.dim_pose = 263
        elif self.dataset_name == 'kit':
            self.dim_pose = 251
        else:
            raise ValueError(f'Unknown dataset name for T2M: {self.dataset_name}')