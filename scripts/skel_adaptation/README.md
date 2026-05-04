# Skeleton Adaptation

Contains scripts to handle skeleton annotation convertions between NTU RGB+D 60/120 and HumanML3D

## Details
Standard formats
* NTU $\rightarrow$ Kinect format (25 joints).
* HumanML3D $\rightarrow$ SMPL format (without hand joints, 22 total).

Some custom formats employed by us:
* Kinect-reduced $\rightarrow$ Kinect format with no hand information (19 joints)


<div align="center">

| NTU Skeleton | SMPL Skeleton |
|:------------:|:-------------:|
| <img src="../../media/ntu_skele.svg" width="40%"> | <img src="../../media/smpl_skele.svg" width="40%"> |

</div>

## Visualization

<br>

You can visually evaluate the correctness of Forward-Backward mapping as well as inspect dataset motion by playing with the `animation.ipynb` script. Outputs will be stored under `media/` folder, prividing for the specified dataset:
* forward mapping
* backward mapping (applied to forward itself)
* the original motion