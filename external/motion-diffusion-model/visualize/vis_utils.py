from model.utils.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl
from data_loaders.unified.scripts.motion_process import recover_from_ric

class npyResults2obj:
    def __init__(self, npy_path, sample_idx, rep_idx, device=0, cuda=True):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True)
        if self.npy_path.endswith('.npz'):
            self.motions = self.motions['arr_0']
        self.motions = self.motions[None][0]
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.rep_idx*self.total_num_samples + self.sample_idx
        self.num_frames = self.motions['motion'][self.absl_idx].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)

        if self.nfeats == 3:
            print(f'Running SMPLify For sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
            motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
            self.motions['motion'] = motion_tensor.cpu().numpy()
        elif self.nfeats == 6:
            self.motions['motion'] = self.motions['motion'][[self.absl_idx]]
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.real_num_frames = self.motions['lengths'][self.absl_idx]

        self.vertices = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)
        self.root_loc = self.motions['motion'][:, -1, :3, :].reshape(1, 1, 3, -1)
        # self.vertices += self.root_loc

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_npy(self, save_path):
        data_dict = {
            'motion': self.motions['motion'][0, :, :, :self.real_num_frames],
            'thetas': self.motions['motion'][0, :-1, :, :self.real_num_frames],
            'root_translation': self.motions['motion'][0, -1, :3, :self.real_num_frames],
            'faces': self.faces,
            'vertices': self.vertices[0, :, :, :self.real_num_frames],
            'text': self.motions['text'][0],
            'length': self.real_num_frames,
        }
        np.save(save_path, data_dict)


class npy2obj:
    """
    Loads a single animation from a .npy file and generates 3D mesh objects.

    This class assumes the input .npy file contains a single NumPy array with the
    shape (Num_frames, 66), representing an animation with 22 joints. It processes
    this joint data to generate a renderable mesh for any given frame.
    """
    def __init__(self, npy_path, device=0, cuda=True):
        """
        Initializes the loader, processes the animation, and prepares for mesh generation.

        Args:
            npy_path (str): The file path to the .npy animation file.
            device (int): The GPU device ID to use for processing.
            cuda (bool): A flag to enable or disable CUDA.
        """
        # --- 1. Load and Reshape Animation Data ---
        print(f"Loading animation from: {npy_path}")
        # Load the raw joint data: expected shape (Num_frames, 66)
        joint_data = np.load(npy_path)

        if joint_data.shape[-1] != 22: # .npy expressed in hml_vec format, convert it back to xyz
            joint_data = np.array(recover_from_ric(torch.tensor(joint_data), 22))

        self.num_frames = joint_data.shape[0]
        
        # Reshape to (Num_frames, num_joints, 3) for the SMPLify process.
        # The number of joints is fixed at 22 as per the requirement.
        try:
            joint_data_reshaped = joint_data.reshape(self.num_frames, 22, 3)
        except ValueError as e:
            print(f"Error: Could not reshape array of shape {joint_data.shape} to ({self.num_frames}, 22, 3).")
            print("Please ensure the input .npy file has a shape of (Num_frames, 66).")
            raise e

        # --- 2. Convert Joint Positions to SMPL Parameters ---
        print(f"Running SMPLify for {self.num_frames} frames. This may take a few moments...")
        j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)
        
        # Convert numpy array to a torch tensor for the model
        motion_tensor, _ = j2s.joint2smpl(torch.from_numpy(joint_data_reshaped).float())
        
        # --- 3. Generate Mesh Vertices from SMPL Parameters ---
        # We use the 'cpu' device for Rotation2xyz as it is lightweight
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        
        # Ensure the motion tensor is on the CPU before generating vertices
        motion_tensor_cpu = motion_tensor.cpu()
        
        # Generate vertices for the entire animation. The output shape will be:
        # (batch_size, num_vertices, 3, num_frames) -> (1, 6890, 3, num_frames)
        self.vertices = self.rot2xyz(motion_tensor_cpu, mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     vertstrans=True)
        print("✅ Processing complete. The class is ready to generate OBJ files.")


    def get_vertices_for_frame(self, frame_index):
        if not 0 <= frame_index < self.num_frames:
            raise IndexError(f"Frame index {frame_index} is out of bounds. Must be between 0 and {self.num_frames - 1}.")
        # The vertices tensor has a batch dimension of 1, so we always index with 0.
        return self.vertices[0, :, :, frame_index].squeeze().tolist()

    def get_trimesh_for_frame(self, frame_index):
        return Trimesh(vertices=self.get_vertices_for_frame(frame_index),
                       faces=self.faces)

    def save_obj(self, save_path, frame_index):
        mesh = self.get_trimesh_for_frame(frame_index)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        print(f"Saved frame {frame_index} to {save_path}")
        return save_path