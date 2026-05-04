"""
Constants relative to
> fixed filenames/paths used in the codebase.
> dataset-specific parameters
"""

# Dataset files formatted as in PySkl toolbox
DATA_FILENAME = {
    'NTU60': 'ntu60_3danno.pkl',
    'NTU120': 'ntu120_3danno.pkl'
}

# Number of classes in each Action Recognition dataset
DATA_NUM_CLASSES = {
    'NTU60': 60,
    'NTU120': 120,
    'NTU-VIBE': 120,
}

# Represent classes we're ignoring in our tests, specifically, those involving >1 skeletons
DATA_IGNORE_CLASSES = {
    'NTU60': [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],  
    'NTU120': [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'NTU-VIBE' : [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
}

# File which contains a mapping from action index to a list of viable captions
# for such action class
NTU_METADATA = "ntu_metadata.json"

# Time Resampling parameters
DATASET_FPS = {
    'NTU60': 30,
    'NTU120': 30,
    'NTU-VIBE': 30,
    'HML3D': 20
}