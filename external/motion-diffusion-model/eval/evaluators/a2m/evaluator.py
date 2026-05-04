from turtle import back
import torch
import os
from os.path import join as pjoin
from model.motion_encoders import STGCN
from model.wrappers import Classifier
from eval.evaluators.config import A2MEvaluatorConfig
import json

from torch import Tensor
from typing import Dict, Tuple

class ActionMotionEvaluator:
    """Action-to-Motion evaluator using ST-GCN model"""
    
    def __init__(self, config: A2MEvaluatorConfig):
        
        self.checkpoints_dir = config.checkpoints_dir
        self.task_split = config.task_split
        self.class_list = config.class_list
        self.fewshot_id = config.fewshot_id
        self.evaluator_type = config.evaluator_type
        self.cfg = config

        self.model = None
        self._pick_evaluator()
        self._setup_model()
    
    def _pick_evaluator(self):
        '''
        Look for an evaluator which was trained on the same class set
        '''
        for name in os.listdir(self.checkpoints_dir):            
            current_path = pjoin(self.checkpoints_dir, name)
            with open(pjoin(current_path, 'info.json'), 'r') as f:
                info = json.load(f)
            if info['task_split'] == self.task_split and info['class_list'] == self.class_list:
                self.checkpoints_dir = pjoin(current_path, 'model_best.pth')
                break
        assert self.checkpoints_dir.endswith('.pth'), f'No matching checkpoint found for : {self.fewshot_id}'
        

    def _setup_model(self):
        """Setup Classifier model for action recognition"""
        # Init. the chosen model
        if self.evaluator_type == 'stgcn':
            model = STGCN(
                dict(layout='humanml', mode='stgcn_spatial'),
                in_channels=3, # xyz format
            )
            margs = {'model': model, 'in_dim': model.out_channels}
        else:
            raise ValueError(f'Unknown evaluator type: {self.evaluator_type}')
        # Wrap it in a classifier
        self.model = Classifier(
            **margs,
            num_classes=self.cfg.num_classes,
        )
        # Load checkpoint
        checkpoint = torch.load(self.checkpoints_dir, map_location=self.cfg.device)
        self.model.load_state_dict(checkpoint)
        print(f"\n>>> Evaluating with [{self.cfg.num_classes}] classes:", self.class_list)
        print(f'>>> A2M Evaluator from: {self.checkpoints_dir}')
        # To device and eval mode
        self.model.to(self.cfg.device)
        self.model.eval()
    
    def get_motion_embeddings(self, *, x: Tensor, y: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Extract motion embeddings"""
        with torch.no_grad():
            features, logits = self.model(x, y)
        return features, logits