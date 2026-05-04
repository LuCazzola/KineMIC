import numpy as np
from scipy.interpolate import interp1d

from typing import Optional

from utils.humanml3d.quaternion import _EPS4, _FLOAT_EPS
from utils.constants.skel import SMPL_DIRECT_MAP, JOINTS_2_DROP, FCOEFF, FLOOR_THRE

EPS = _FLOAT_EPS # _FLOAT_EPS, _EPS4

def resample_motion(motion: np.ndarray, original_fps: int = 30, target_fps: int = 20) -> np.ndarray:
    """
    Subsample motion to target_fps.
    """
    assert motion.ndim == 3 # (T, J, D)

    T = motion.shape[0]
    duration = (T - 1) / original_fps
    original_times = np.linspace(0, duration, T)
    target_T = int(duration * target_fps) + 1
    target_times = np.linspace(0, duration, target_T)
    interp_fn = interp1d(original_times, motion, axis=0, kind='linear')
    return interp_fn(target_times).astype(motion.dtype)

def forward_map(ntu_joints: np.ndarray):
    """
    Map NTU joints -> SMPL joints.
    Returns: (T, 22, 3) joints: HumanML3D compatible skeletons (SMPL).
    """
    assert ntu_joints.ndim == 3 # (T, J, D)

    T = ntu_joints.shape[0]
    ntu_joints = ntu_joints.astype(np.float32)  # Convert to float32 for consistency

    smpl_joints = np.zeros((T, 22, 3), dtype=np.float32)
    for ntu_idx, smpl_idx in SMPL_DIRECT_MAP["kinect"].items():
        smpl_joints[:, smpl_idx, :] = ntu_joints[:, ntu_idx, :]

    spineBase = ntu_joints[:, 0, :]
    spineMid  = ntu_joints[:, 1, :]
    spineShoulder = ntu_joints[:, 20, :]
    leftShoulder = ntu_joints[:, 4, :]
    rightShoulder = ntu_joints[:, 8, :]
    
    # approximate "forward" = up × across
    torso_vec = spineShoulder - spineBase
    torso_dir = torso_vec / (np.linalg.norm(torso_vec, axis=1, keepdims=True) + EPS)
    across = rightShoulder - leftShoulder  # side-to-side (X-axis)
    across = across / (np.linalg.norm(across, axis=1, keepdims=True) + EPS)
    forward_dir = np.cross(torso_dir, across)
    forward_dir = forward_dir / (np.linalg.norm(forward_dir, axis=1, keepdims=True) + EPS)

    # Flip correction (when forward direction is suddenly inverted it's likely that motion is degenerated)
    # So we assume that it should be corrected (otherwise the spine curve is reversed)
    #flipped = []
    #for i in range(1, T):
    #    if not (-0.95 < np.dot(forward_dir[i-1], forward_dir[i]) < 0.95):
    #        forward_dir[i] *= -1
    #        flipped.append(i)
    #if len(flipped) > 0:
    #    print("Flip occured: ", flipped)

    # Clavicles
    smpl_joints[:, 13, :] = spineShoulder + (leftShoulder - spineShoulder) * FCOEFF.clavicle_offset
    smpl_joints[:, 14, :] = spineShoulder + (rightShoulder - spineShoulder) * FCOEFF.clavicle_offset
    # Spine3
    smpl_joints[:, 9, :] = (spineMid + spineShoulder) / 2.0
    
    # Spine1
    smpl_joints[:, 3, :] = (spineMid + spineBase) / 2.0 \
        + FCOEFF.spine1_curve * (-forward_dir) + FCOEFF.spine1_curve * (-torso_dir)
    
    # Spine2
    smpl_joints[:, 6, :] = smpl_joints[:, 9, :] + (smpl_joints[:, 3, :] - smpl_joints[:, 9, :]) * FCOEFF.spine2_offset \
        + FCOEFF.spine2_curve * (-forward_dir) + FCOEFF.spine2_curve * (-torso_dir)

    return smpl_joints

def backward_map(smpl_joints: np.ndarray):
    """
    Map NTU joints -> SMPL joints.
    Returns : (T, 19, 3) as hand-related joints are dropped
    """
    T = smpl_joints.shape[0]
    source_dtype = smpl_joints.dtype
    smpl_joints = smpl_joints.astype(np.float32)  # Convert to float32 for consistency

    ntu_joints = np.zeros((T, 25, 3), dtype=np.float32)
    # Reverse the direct mapping (collar-bone is implicitly dropped)
    smpl_to_ntu_map = {v: k for k, v in SMPL_DIRECT_MAP["kinect"].items() if k not in JOINTS_2_DROP["kinect"] and k != 1}
    for smpl_idx, ntu_idx in smpl_to_ntu_map.items():
        ntu_joints[:, ntu_idx, :] = smpl_joints[:, smpl_idx, :]

    pelvis = smpl_joints[:, 0, :]
    spine1 = smpl_joints[:, 3, :]
    spine3 = smpl_joints[:, 9, :]

    # Project spine1 onto the torso vector and use it to compute the mid spine joint
    torso_vec = spine3 - pelvis
    torso_dir = torso_vec / (np.linalg.norm(torso_vec, axis=1, keepdims=True) + EPS)
    spine1_proj = pelvis + np.sum((spine1 - pelvis) * torso_dir, axis=1, keepdims=True) * torso_dir
    ntu_joints[:, 1, :] = (spine1_proj + spine3) / 2.0
    # SpineShoulder (algebra inverse of forward_map)
    ntu_joints[:, 20, :] = 2.0*spine3 - ntu_joints[:, 1, :]
    
    # Drop hand joints
    ntu_joints = np.delete(ntu_joints, list(JOINTS_2_DROP["kinect"]), axis=1) # new shape (T, 19, 3)

    return ntu_joints.astype(source_dtype)  # Convert back to original dtype

def align_motion(joints: np.ndarray, displacement: Optional[np.ndarray] = None):
    """
    joints : (T, J, D)
    displacement : (D,) or None
    
    Returns: Aligns the motion by shifting the root joint to the origin and the floor to Y=0.
    NOTE: adding displacement is done anyway when computing the redundant motion features.
    """
    assert displacement is None or displacement.shape[0] == joints.shape[2], "Displacement must be None or have shape (D,)"

    if displacement is None:
        root_x, root_z = joints[0, 0, 0], joints[0, 0, 2] # XZ (horizontal plane) of root joint
        j_heights = np.sort(joints[:, :, 1].flatten())
        floor_y = j_heights[:FLOOR_THRE].mean() # Y (vertical) of the floor, computed as the mean of the lowest FLOOR_THRE joint heights
        displacement = np.array([root_x, floor_y, root_z])  # (X, Y, Z) displacement vector

    joints -= displacement

    return joints, displacement