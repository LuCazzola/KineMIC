# Exploratory Data Analysis (EDA) Toolkit

This directory contains a comprehensive suite of tools for analyzing motion datasets, specifically designed for comparing NTU RGB+D and HumanML3D datasets. The toolkit provides semantic, statistical, and visualization analysis capabilities.

```
scripts/eda/
├── semantic/           # Semantic similarity analysis
├── statistical/        # Statistical comparison tools  
└── viz/               # Motion visualization utilities
```

## Analysis Components

### 🔍 Semantic Analysis (`semantic/`)

**Purpose**: Analyze semantic relationships between motion datasets using CLIP embeddings and k-nearest neighbor analysis.

**Key Features**:
- Compute CLIP text embeddings for motion descriptions
- K-NN density analysis between NTU action labels and HumanML3D descriptions
- Semantic similarity mapping and clustering
- Support for both raw action labels and natural language descriptions

**Usage**:
```bash
python -m scripts.eda.semantic.rank --k 200
```

**Outputs**:
- JSON files with k-NN statistics and closest matches
- Density scores for action-description alignment
- Cached embeddings for efficient recomputation

### 📊 Statistical Analysis (`statistical/`)

**Purpose**: Comprehensive statistical comparison between motion datasets including distribution analysis, outlier detection, and dimensionality reduction.

**Key Features**:
- **Sequence Analysis**: Length distributions, statistical tests (KS-test)
- **Feature Analysis**: Global statistics (mean, std, min/max) across all features
- **Motion Dynamics**: Velocity, acceleration, and jerk analysis for motion quality assessment
- **Outlier Detection**: IQR-based outlier identification with configurable thresholds
- **Dimensionality Analysis**: Joint PCA analysis with 2D/3D visualizations

**Usage**:
```bash
# Basic XYZ coordinate analysis
python -m scripts.eda.statistical.run_analysis --data-rep xyz --use-cache

# HumanML3D vector format analysis
python scripts.eda.statistical.run_analysis --data-rep hml_vec --outlier-cutoff 1.5

# Analyze specific subsets
python scripts.eda.statistical.run_analysis --data-rep xyz --ntu-set ntu_subset.txt --hml-set hml_subset.txt
```

**Outputs**:
- Comprehensive plots: sequence lengths, feature statistics, motion dynamics, PCA analysis
- Statistical test results and significance tests
- Outlier detection with blacklists for problematic sequences
- Summary logs with detailed comparison metrics

### 🎬 Visualization (`viz/`)

**Purpose**: Generate 3D motion visualizations for qualitative analysis and presentation.

**Key Features**:
- **3D Motion Rendering**: High-quality 3D skeleton animations
- **Multi-format Support**: Both Kinect (25 joints) and SMPL (22 joints) skeletons
- **Side-by-side Comparisons**: Synchronized visualization of different representations
- **Batch Processing**: Efficient generation of multiple animation samples
- **Customizable Output**: Configurable frame rates, styling, and camera angles

**Usage**:
```bash
# Generate side-by-side comparisons
python -m scripts.eda.viz.make_animation --samples samples.txt --first_n 10

# Include raw Kinect visualizations
python -m scripts.eda.viz.make_animation --samples samples.txt --raw-kinect
```

**Outputs**:
- MP4 video files with 3D motion animations
- Side-by-side comparison videos (Kinect vs SMPL)
- Raw Kinect motion visualizations
- Batch video compilations