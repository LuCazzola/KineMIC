import torch
from typing import Union
from .t2m.evaluator import TextMotionEvaluator
from .a2m.evaluator import ActionMotionEvaluator
from .config import T2MEvaluatorConfig, A2MEvaluatorConfig

class EvaluatorWrapper:
    """Main evaluator wrapper that handles different dataset types"""
    
    def __init__(self, dataset_name: str, device: torch.device, **kwargs):
        self.dataset_name = dataset_name
        self.device = device
        self.evaluator = self._create_evaluator(**kwargs)
    
    def _create_evaluator(self, **kwargs) -> Union[TextMotionEvaluator, ActionMotionEvaluator]:
        """Create appropriate evaluator based on dataset and conditioning mode"""
        if self.dataset_name in ['humanml', 'kit']:
            config = T2MEvaluatorConfig(
                dataset_name=self.dataset_name,
                device=self.device,
                **kwargs
            )
            return TextMotionEvaluator(config)
            
        elif self.dataset_name in ['ntu60', 'ntu120', 'ntu-vibe']:
            config = A2MEvaluatorConfig(
                dataset_name=self.dataset_name,
                device=self.device,
                **kwargs
            )
            return ActionMotionEvaluator(config)
            
        else:
            raise ValueError(f'Unknown dataset name: {self.dataset_name}')
    
    def get_motion_embeddings(self, *kargs, **kwargs):
        """Get motion embeddings"""
        return self.evaluator.get_motion_embeddings(*kargs, **kwargs)
    
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        """Get co-embeddings for text and motion"""
        if not isinstance(self.evaluator, TextMotionEvaluator):
            raise ValueError("Co-embeddings only available for text-motion evaluators")
        
        return self.evaluator.get_co_embeddings(word_embs, pos_ohot, cap_lens, motions, m_lens)