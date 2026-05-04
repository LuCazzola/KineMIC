from diffusion.nn import mean_flat, sum_flat
from torch.nn import functional as F
import torch
import numpy as np
import enum

# MAE (L1)
def angle_l1(angle1, angle2):
    a = angle1 - angle2
    a = (a + (torch.pi/2)) % torch.pi - (torch.pi/2)
    return torch.abs(a)

def diff_l1(a, b):
    return torch.abs(a - b)

# MSE (L2)
def angle_l2(angle1, angle2):
    a = angle1 - angle2
    a = (a + (torch.pi/2)) % torch.pi - (torch.pi/2)
    return a ** 2

def diff_l2(a, b):
    return (a - b) ** 2

def contrastive_loss(a, b, logit_scale, positive_mask=None):
    """ Standard contrastive loss, similar to CLIP """
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    logits_a = logit_scale.exp() * a @ b.t()
    logits_b = logits_a.t()
    labels = torch.arange(a.shape[0], device=a.device)
    return (F.cross_entropy(logits_a, labels) + F.cross_entropy(logits_b, labels)) / 2

def soft_nearest_neighbors_loss_simple(a, b, y_a, y_b, logit_scale):
    """ Soft-Nearest Neighbors loss (naive implementation) """
    a, b = F.normalize(a, p=2, dim=-1), F.normalize(b, p=2, dim=-1)
    logits = logit_scale.exp() * a @ b.t()  # cosine sime
    positive_mask = torch.eq(y_a.unsqueeze(1), y_b.unsqueeze(0)) # positive matchings mask
    numerator = torch.sum(positive_mask.float() * torch.exp(logits), dim=1) # Positive similarities
    denominator = torch.sum(torch.exp(logits), dim=1)  # All similarities
    return -torch.log(numerator / (denominator + 1e-8))

def soft_nearest_neighbors_loss(a, b, y_a, y_b, logit_scale):
    """ Soft-Nearest Neighbors loss (More numerically stable) """
    a, b = F.normalize(a, p=2, dim=-1), F.normalize(b, p=2, dim=-1) # Normalize
    logits = logit_scale.exp() * a @ b.t() # cosine sim
    mask = torch.eq(y_a.unsqueeze(1), y_b.unsqueeze(0)) # positive matchings mask
    # logsumexp for stability
    neg_inf = torch.tensor(-float('inf'), dtype=logits.dtype, device=logits.device)
    log_positives = torch.logsumexp(torch.where(mask, logits, neg_inf), dim=1) # numerator
    log_all = torch.logsumexp(logits, dim=1) # denominator
    return -(log_positives - log_all) # difference, no division (we're in log space)

def entropy_regularization(x):
    x_norm = F.normalize(x, p=2, dim=-1)
    similarities = torch.matmul(x_norm, x_norm.t())
    probs = F.softmax(similarities, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1) # Compute entropy
     

class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    MAE = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

def get_geom_loss(loss_type, prefix='diff'):
    assert prefix in ['diff', 'angle']
    if prefix == 'diff': 
        if loss_type == LossType.MAE:
            return diff_l1
        elif loss_type == LossType.MSE:
            return diff_l2
        else :
            raise ValueError("Unsupported loss type: {}".format(loss_type))
    elif prefix == 'angle':
        if loss_type == LossType.MAE:
            return angle_l1
        elif loss_type == LossType.MSE:
            return angle_l2
        else :
            raise ValueError("Unsupported loss type: {}".format(loss_type))

def get_main_loss(loss_type):
    if loss_type == 'MSE' :
        return LossType.MSE
    elif loss_type == 'MAE' :
        return LossType.MAE
    else :
        raise ValueError("Unsupported loss type: {}".format(loss_type))


def masked_lx(a, b, mask, loss_fn=diff_l2, epsilon=1e-8, entries_norm=True):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen
    loss = loss_fn(a, b)
    loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
    n_entries = a.shape[1]
    if len(a.shape) > 3:
        n_entries *= a.shape[2]
    non_zero_elements = sum_flat(mask)
    if entries_norm:
        # In cases the mask is per frame, and not specifying the number of entries per frame, this normalization is needed,
        # Otherwise set it to False
        non_zero_elements *= n_entries
    # print('mask', mask.shape)
    # print('non_zero_elements', non_zero_elements)
    # print('loss', loss)
    loss_val = loss / (non_zero_elements + epsilon)  # Add epsilon to avoid division by zero
    # print('mse_loss_val', mse_loss_val)
    return loss_val


def masked_goal_lx(pred_goal, ref_goal, cond, all_goal_joint_names, diff_loss=diff_l2, angle_loss=angle_l2):
    all_goal_joint_names_w_traj = np.append(all_goal_joint_names, 'traj')
    target_joint_idx = [[np.where(all_goal_joint_names_w_traj == j)[0][0] for j in sample_joints] for sample_joints in cond['target_joint_names']]
    loc_mask = torch.zeros_like(pred_goal[:,:-1], dtype=torch.bool)
    for sample_idx in range(loc_mask.shape[0]):
        loc_mask[sample_idx, target_joint_idx[sample_idx]] = True
    loc_mask[:, -1, 1] = False  # vertical joint of 'traj' is always masked out
    loc_loss = masked_lx(pred_goal[:,:-1], ref_goal[:,:-1], loc_mask, loss_fn=diff_loss, entries_norm=False)
    
    heading_loss = masked_lx(pred_goal[:,-1:, :1], ref_goal[:,-1:, :1], cond['is_heading'].unsqueeze(1).unsqueeze(1), loss_fn=angle_loss, entries_norm=False)

    loss =  loc_loss + heading_loss
    return loss
