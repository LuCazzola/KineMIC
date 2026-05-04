
import numpy as np
import random
import torch
from einops import rearrange

class MotionDataAugmenter():
    def __init__(self, opt, transforms=[]):
        self.opt = opt
        self.transforms = transforms

    def set_transforms(self, transforms):
        self.transforms = transforms

    def __call__(self, motion, m_length):
        for transform in self.transforms:
            motion = transform(motion, m_length)
            m_length = motion.shape[0] # Update length if changed
        return motion

    def to_unit_length(self, motion, m_length):
        """
        Crop the motions in to times of unit_length, and introduce small variations
        The cropped length becomes multiples of unit_length
        """
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        
        #original_length = None
        if self.opt.fixed_len > 0:
            # Crop fixed_len
            #original_length = m_length
            m_length = self.opt.fixed_len
        
        idx = random.randint(0, len(motion) - m_length)
        return motion[idx:idx+m_length]
        

    def random_rot(self, motion, m_length, angle_range=(-np.pi/6, np.pi/6)):
        """
        Apply a random rotation to motion data around the root joint of the first frame.
        This ensures the character rotates in place without translational artifacts.
        """
        # Reshape to a more convenient (Time, Joints, XYZ) format
        motion_reshaped = rearrange(motion, 't (j xyz) -> t j xyz', xyz=3)

        # 1. Define the pivot point for rotation.
        # We use the position of the root joint (joint 0) in the first frame.
        pivot = motion_reshaped[0, 0, :]

        # 2. Center the entire motion sequence around this pivot point.
        # This is done by subtracting the pivot's coordinates from every joint in every frame.
        centered_motion = motion_reshaped - pivot

        # 3. Create a single random 3D rotation matrix.
        rot_x = np.random.uniform(angle_range[0], angle_range[1])
        rot_y = np.random.uniform(angle_range[0], angle_range[1])
        rot_z = np.random.uniform(angle_range[0], angle_range[1])

        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(rot_x), -np.sin(rot_x)],
            [0, np.sin(rot_x), np.cos(rot_x)]
        ])
        R_y = np.array([
            [np.cos(rot_y), 0, np.sin(rot_y)],
            [0, 1, 0],
            [-np.sin(rot_y), 0, np.cos(rot_y)]
        ])
        R_z = np.array([
            [np.cos(rot_z), -np.sin(rot_z), 0],
            [np.sin(rot_z), np.cos(rot_z), 0],
            [0, 0, 1]
        ])
        R = R_z @ R_y @ R_x

        # 4. Apply this rotation to the centered motion.
        # np.einsum efficiently multiplies the rotation matrix with each joint's XYZ vector.
        rotated_centered_motion = np.einsum('ij,tkj->tki', R, centered_motion)

        # 5. Translate the rotated motion back to its original position.
        final_motion = rotated_centered_motion + pivot

        # Reshape back to the original (T, J*3) format and return
        return rearrange(final_motion, 't j xyz -> t (j xyz)')
    
    def random_scale(self, motion, m_length, scale_range=(0.9, 1.1)):
        """Apply Random scaling"""
        return motion * np.random.uniform(scale_range[0], scale_range[1])
    
    def gaussian_noise(self, motion, m_length, mean=0.0, std=0.05):
        """
        Injects Gaussian noise into the motion data. This can help the model
        become more robust to small variations in joint positions.
        """
        return motion + np.random.normal(loc=mean, scale=std, size=motion.shape)
