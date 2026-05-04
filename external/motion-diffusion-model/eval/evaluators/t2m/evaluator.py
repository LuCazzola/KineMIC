
import torch
import numpy as np

from typing import Tuple
from pathlib import Path

from .modules import MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo
from ..config import T2MEvaluatorConfig


class TextMotionEvaluator:
    """Text-to-Motion evaluator using BiGRU co-embedding model"""
    
    def __init__(self, config: T2MEvaluatorConfig):
        self.config = config
        self.device = config.device
        
        self.text_encoder = None
        self.motion_encoder = None
        self.movement_encoder = None
        
        self._setup_models()
    
    def _setup_models(self):
        """Initialize and load BiGRU models"""

        self.movement_encoder = MovementConvEncoder(
            self.config.dim_pose - 4,
            self.config.dim_movement_enc_hidden,
            self.config.dim_movement_latent
        )
        
        self.text_encoder = TextEncoderBiGRUCo(
            word_size=self.config.dim_word,
            pos_size=self.config.dim_pos_ohot,
            hidden_size=self.config.dim_text_hidden,
            output_size=self.config.dim_coemb_hidden,
            device=self.config.device
        )
        
        self.motion_encoder = MotionEncoderBiGRUCo(
            input_size=self.config.dim_movement_latent,
            hidden_size=self.config.dim_motion_hidden,
            output_size=self.config.dim_coemb_hidden,
            device=self.config.device
        )
        
        # Load checkpoint and move to device
        self._load_checkpoint()
        self._to_device_and_eval()
    
    def _load_checkpoint(self):
        """Load pre-trained weights"""
        # Determine checkpoint directory
        ckpt_dir = 't2m' if self.config.dataset_name == 'humanml' else 'kit'
        checkpoint_path = Path(self.config.checkpoints_dir) / ckpt_dir / 'text_mot_match' / 'model' / 'finest.tar'
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.movement_encoder.load_state_dict(checkpoint['movement_encoder'])
        self.motion_encoder.load_state_dict(checkpoint['motion_encoder'])
        if self.text_encoder:
            self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        
        print(f'Loading T2M Evaluation Model (Epoch {checkpoint["epoch"]}) Completed!!')
    
    def _to_device_and_eval(self):
        """Move models to device and set eval mode"""
        self.movement_encoder.to(self.device).eval()
        self.motion_encoder.to(self.device).eval()
        if self.text_encoder:
            self.text_encoder.to(self.device).eval()
    
    def _prepare_motions(self, motions: torch.Tensor, m_lens: torch.Tensor):
        """Prepare motion data for processing"""
        motions = motions.detach().to(self.device).float()
        
        # Sort by length for efficient processing
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        
        return motions, m_lens, align_idx
    
    def get_motion_embeddings(self, motions: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        """Extract motion embeddings"""
        with torch.no_grad():
            motions, m_lens, _ = self._prepare_motions(motions, m_lens)
            
            # Movement encoding (remove last 4 dimensions)
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.config.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
            
        return motion_embedding
    
    def get_co_embeddings(self, word_embs: torch.Tensor, pos_ohot: torch.Tensor, 
                         cap_lens: torch.Tensor, motions: torch.Tensor, 
                         m_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract co-embeddings for text and motion"""
        if self.text_encoder is None:
            raise ValueError("Text encoder is not available in the current evaluator.")
        
        with torch.no_grad():
            # Prepare text data
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            
            # Prepare motion data
            motions, m_lens, align_idx = self._prepare_motions(motions, m_lens)
            
            # Motion encoding
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.config.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
            
            # Text encoding
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
            
        return text_embedding, motion_embedding
