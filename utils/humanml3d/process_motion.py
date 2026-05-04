"""
Credits: This code is motly taken from
1. https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb
2. https://github.com/EricGuo5513/HumanML3D/blob/main/cal_mean_variance.ipynb

Edits:
+   floor is not detected as the hard minimum height of all joints in an animation.
    instead, due to noise in the data, it is averaged over 'floor_thre' lowest joints (in height)
+   general modifications to align with the rest of the codebase
"""
import torch
import numpy as np
from types import SimpleNamespace

from .skeleton import Skeleton
from .quaternion import *
from .paramUtil import t2m_kinematic_chain, t2m_raw_offsets

from utils.constants.skel import SKEL_INFO
SKL = SKEL_INFO['smpl']


"""
DATA PROCESSING
"""

def uniform_skeleton(positions, opt):
    src_skel = Skeleton(opt.n_raw_offsets, opt.kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = opt.target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[SKL.l_idx1]).max() + np.abs(src_offset[SKL.l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[SKL.l_idx1]).max() + np.abs(tgt_offset[SKL.l_idx2]).max()
    scale_rt = tgt_leg_len / src_leg_len

    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, SKL.face_joint_indx)

    '''Forward Kinematics'''
    src_skel.set_offset(opt.target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)

    return new_joints

def globalize_pos(positions, skeleton, floor_thre, mirror=False):
    '''Put on Floor'''
    # Floor is not detected as the hard minimum, but as
    # the average of the lowest Y across floor_thres joints (for stability to noise)
    j_heights = np.sort(positions[:, :, 1].flatten())
    floor_height = j_heights[:floor_thre].mean()
    positions[:, :, 1] -= floor_height
    #     print(floor_height)
    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = skeleton.face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    if mirror:
        across *= -1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    
    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    return positions


def process_file(positions, floor_thre, feet_thre, opt, mirror=False):
    """
    Process a raw motion file (T,J,3) and returns HML feat. vec.
    redundant representation (T-1, 263).
    """
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, opt)    

    '''Put on Floor'''
    # Floor is not detected as the hard minimum, but as
    # the average of the lowest Y across floor_thres joints (for stability to noise)
    j_heights = np.sort(positions[:, :, 1].flatten())
    floor_height = j_heights[:floor_thre].mean()
    positions[:, :, 1] -= floor_height
    #     print(floor_height)
    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = SKL.face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, SKL.fid_l, 0] - positions[:-1, SKL.fid_l, 0]) ** 2
        feet_l_y = (positions[1:, SKL.fid_l, 1] - positions[:-1, SKL.fid_l, 1]) ** 2
        feet_l_z = (positions[1:, SKL.fid_l, 2] - positions[:-1, SKL.fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, SKL.fid_r, 0] - positions[:-1, SKL.fid_r, 0]) ** 2
        feet_r_y = (positions[1:, SKL.fid_r, 1] - positions[:-1, SKL.fid_r, 1]) ** 2
        feet_r_z = (positions[1:, SKL.fid_r, 2] - positions[:-1, SKL.fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions, opt):
        skel = Skeleton(opt.n_raw_offsets, opt.kinematic_chain, 'cpu')
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, SKL.face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions, opt):
        skel = Skeleton(opt.n_raw_offsets, opt.kinematic_chain, 'cpu')
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, SKL.face_joint_indx, smooth_forward=True)
        
        # FIXME (do i need this?)
        #quat_params = qfix(quat_params)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot
    
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions, opt)
    positions = get_rifke(positions)
    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    """
    Recover root rotation and position from HML vec.
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    """
    Recover global joint positions from rotation data in HML vec. by applying forward kinematics.
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions

def recover_from_ric(data, joints_num):
    """
    Recover global joint positions from HML vec.
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

def recover_velocities(hml_features, joints_num=22):
    """
    Extract velocity components from HML redundant feature representation.
    
    Args:
        hml_features: (T, 263) or (T, 256) HML feature vector 
        joints_num: Number of joints (default 22 for SMPL)
    
    Returns:
        root_velocity: (T, 2) - XZ linear velocity of root joint
        joint_velocities: (T, joints_num*3) - 3D velocities of all joints
    """
    # Extract root linear velocity (XZ plane)
    root_velocity = hml_features[:, 1:3]  # (T, 2)
    
    # Extract joint velocities 
    local_vel_start = 4 + (joints_num - 1) * 9
    local_vel_end = local_vel_start + joints_num * 3
    joint_velocities = hml_features[:, local_vel_start:local_vel_end]  # (T, joints_num*3)
    
    return root_velocity, joint_velocities

# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def cal_mean_variance(data, joints_num):
    """
    Compute mean and variance for a HML feat. vec.
    """
    #data = np.concatenate(data_list, axis=0)

    Mean = data.mean(axis=0)
    Std = data.std(axis=0)

    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    return Mean, Std

"""
WRAPPERS
"""

def motion_2_hml_vec(
        motion_seq: np.ndarray,
        floor_thre: int = 1,
        feet_thre: float = 0.002,
        mirror: bool = False
) -> np.ndarray:
    """
    Wrapper function to convert a raw 3-D joint sequence into the 263-D redundant representation.

    Parameters
    ----------
    motion_seq : np.ndarray
        Shape (T, J, 3).  J must match the HumanML3D skeleton (22 joints - SMPL).
    floor_thre : int, optional
        Number of joints to consider for floor detection.  Default is 1, which
        means the lowest (Y-axis) joint is used to detect the floor.
    feet_thre : float, optional
        Foot-contact velocity threshold.  Default is 0.002
        (same as the original script).
    mirror : bool, optional
        Whether to randomly mirror the motion sequence along the X-axis.  Default is False.
    Returns
    -------
    np.ndarray
        Shape (T-1, 263). Redundant representation feature vector suitable for training Motion generative models.
    """
    assert motion_seq.ndim == 3 and motion_seq.shape[2] == 3, "Input must be (T, J, 3)."
    
    motion_seq = motion_seq.astype(np.float32)
    # NOTE: code might be extended to allow different skeletons in the future.
    # for now always assume HumanML3D skeleton (SMPL) with 22 joints.
    opt = SimpleNamespace(
        n_raw_offsets=None,
        kinematic_chain=None,
        target_offset=None
    )

    opt.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    opt.kinematic_chain = t2m_kinematic_chain
    # Target offsets are determined from the first pose
    tgt_skel = Skeleton(opt.n_raw_offsets, opt.kinematic_chain, 'cpu')
    opt.target_offset = tgt_skel.get_offsets_joints(torch.from_numpy(motion_seq[0])) # Offsets based on frame 0
    # Process
    features, *_ = process_file(motion_seq, floor_thre, feet_thre, opt, mirror=mirror)   # (T-1, 263)
    
    return features.astype(np.float32)