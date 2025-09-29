# Sensor-Invariant Tactile Representation (SITR)

This is the official codebase for the paper "Sensor-Invariant Tactile Representation" (ICLR 2025). For more details, visit our [project website](https://hgupt3.github.io/sitr/) or read the [arXiv paper](https://arxiv.org/abs/2502.19638).

For instructions on how to generate your own simulation dataset, you can visit [Generation Codebase](https://github.com/hgupt3/gs_blender).

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets and Weights](#datasets-and-weights)
- [Usage](#usage)
  - [Classification](#classification)
  - [Pose Estimation](#pose-estimation)
- [Data Structure and Loading](#data-structure-and-loading)
  - [Directory Structures](#directory-structures)
  - [Data Loader Classes](#data-loader-classes)
  - [Data Preprocessing](#data-preprocessing)
  - [Using Your Own Data](#using-your-own-data)
- [Network Architectures](#network-architectures)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## Overview

This project implements SITR (Sensor-Invariant Tactile Representations), a novel method for extracting sensor-invariant representations that enables zero-shot transfer across optical tactile sensors. Our approach utilizes a transformer-based architecture trained on a diverse dataset of simulated sensor designs, allowing it to generalize to new sensors in the real world with minimal calibration.

The codebase provides implementations for:
- Base representation learning using a transformer architecture with supervised contrastive learning
- Object classification across multiple sensor designs (GelSight Mini [4 variations], DIGIT, Hex, Wedge)
- Pose estimation across multiple sensor designs (GelSight Mini, DIGIT, Hex, Wedge)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/hgupt3/gsrl.git
   cd gsrl
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Quick Start

To quickly get started with SITR:

1. Download the pre-trained weights and dataset from our [HuggingFace dataset](https://huggingface.co/datasets/hgupt3/sitr_dataset). Instructions are present there.
2. Run the classification demo:
   ```
   python plot_classifier.py --load your_path/checkpoints/classification/ --val-path your_path/classification_dataset/
   ```
3. Run the pose estimation demo:
   ```
   python plot_pose_estimater.py --load checkpoints/pose_estimation/ --val-path your_path/pose_dataset/
   ```

## Usage

### Classification

Evaluate and visualize classification results:

```
python plot_classifier.py
```

Arguments:
- `--base_model`: Model architecture (default: 'SITR_base')
- `--load`: Path to load model weights from (default: 'checkpoints/classification/')
- `--batch-size`: Batch size for evaluation (default: 16)
- `--amp`: Enable mixed precision training (default: True)
- `--calibration-config`: Number of calibration images to use (0, 4, 8, 9, or 18) (default: 18)
- `--device`: Device to run evaluation on (default: 'cuda:0')
- `--val-path`: Path to validation dataset (default: 'classification_dataset/val_set')

### Pose Estimation

Evaluate and visualize pose estimation results:

```
python plot_pose_estimater.py
```

Arguments:
- `--base_model`: Model architecture (default: 'SITR_base')
- `--load`: Path to load model weights from (default: 'checkpoints/pose_estimation/')
- `--batch-size`: Batch size for evaluation (default: 32)
- `--amp`: Enable mixed precision training (default: True)
- `--calibration-config`: Number of calibration images to use (0, 4, 8, 9, or 18) (default: 18)
- `--device`: Device to run evaluation on (default: 'cuda:0')
- `--val-path`: Path to validation dataset (default: 'pose_dataset/val_set')

## Data Structure and Loading

The codebase provides data loader classes for different tasks. Each data loader expects a specific data structure and supports various configuration options.

### Directory Structures

#### 1. Base Representation Learning (sim_dataset)
```
data_root/
├── sensor_0000/
│   ├── calibration/          # Calibration images
│   │   ├── 0000.png         # Background image
│   │   ├── 0001.png         # Calibration image 1
│   │   └── ...              # More calibration images
│   ├── samples/             # Sample images
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   ├── dmaps/               # Depth maps (optional)
│   │   ├── 0000.npy
│   │   └── ...
│   └── norms/               # Surface normals (optional)
│       ├── 0000.npy
│       └── ...
├── sensor_0001/
└── ...
```

#### 2. Classification (classification_dataset)
```
data_root/
├── sensor_0000/
│   ├── calibration/          # Same as above
│   ├── samples/             # Organized by class
│   │   ├── class_0000/      # Class 0 samples
│   │   │   ├── 0000.png
│   │   │   └── ...
│   │   ├── class_0001/      # Class 1 samples
│   │   │   ├── 0000.png
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── sensor_0001/
└── ...
```

#### 3. Pose Estimation (pose_dataset)
```
data_root/
├── sensor_0000/
│   ├── calibration/          # Same as above
│   ├── samples/             # Sample images
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   └── locations/           # Location data for pose estimation
│       ├── 0000.npy
│       ├── 0001.npy
│       └── ...
├── sensor_0001/
└── ...
```

### Data Preprocessing

The data loaders automatically handle:
- Background subtraction using calibration images
- Image normalization using pre-computed statistics

### Data Loader Classes

1. **sim_dataset**: Base dataset for representation learning
   - Supports contrastive learning with paired samples
   - Handles calibration images and background subtraction
   - Configuration options:
     - `calibration_config`: Number of calibration images (0, 4, 8, 9, or 18)
     - `augment`: Enable data augmentation
     - `sendTwo`: Enable contrastive learning with paired samples
     - `num_samples`: Number of samples to use (optional)
     - `num_sensors`: Number of sensors to use (optional)

2. **classification_dataset**: For object classification tasks
   - Supports multiple sensor configurations
   - Handles class labels and sensor IDs
   - Configuration options:
     - `sensor_list`: List of sensor IDs to use
     - `class_list`: List of class IDs to include
     - `calibration_config`: Number of calibration images
     - `augment`: Enable data augmentation

3. **pose_dataset**: For pose estimation tasks
   - Supports sequential and random sample pairs
   - Handles location data for pose estimation
   - Configuration options:
     - `sensor_idx`: Specific sensor to use
     - `random_final`: Use random or sequential final states
     - `augment`: Enable data augmentation

### Using Your Own Data

To use the data loaders with your own data:

1. Organize your data following the appropriate directory structure above
2. Ensure all images (calibration and samples) are:
   - Resolution: 224x224 pixels
   - Ideally cropped around the center of the sensor for best results
3. Ensure calibration images are properly named and organized as follows:

   #### Calibration Image Layout
   ```
   Background: 0000.png (sensor without any object)

   Sphere Calibration (4mm diameter sphere):
   ┌────────────┬────────────┬────────────┐
   │  0001.png  │  0002.png  │  0003.png  │
   │  Top Left  │ Top Middle │  Top Right │
   ├────────────┼────────────┼────────────┤
   │  0004.png  │  0005.png  │  0006.png  │
   │  Mid Left  │   Center   │  Mid Right │
   ├────────────┼────────────┼────────────┤
   │  0007.png  │  0008.png  │  0009.png  │
   │  Bot Left  │ Bot Middle │  Bot Right │
   └────────────┴────────────┴────────────┘

   Cube Corner Calibration (any reasonable cube):
   ┌────────────┬────────────┬────────────┐
   │  0010.png  │  0011.png  │  0012.png  │
   │  Top Left  │ Top Middle │  Top Right │
   ├────────────┼────────────┼────────────┤
   │  0013.png  │  0014.png  │  0015.png  │
   │  Mid Left  │   Center   │  Mid Right │
   ├────────────┼────────────┼────────────┤
   │  0016.png  │  0017.png  │  0018.png  │
   │  Bot Left  │ Bot Middle │  Bot Right │
   └────────────┴────────────┴────────────┘
   ```

   For each calibration image, place the object (sphere or cube) roughly indented in the corresponding region of the sensor. For example, in `0001.png`, the sphere should be indented in the top-left region.

4. Configure the appropriate data loader with your sensor IDs or class labels

Examples:

```python
# Base representation learning
from dataloaders import sim_dataset
dataset = sim_dataset(
    path='your_data_path',
    augment=True,
    calibration_config=18,
    sendTwo=True  # for contrastive learning
)

# Classification
from dataloaders import classification_dataset
dataset = classification_dataset(
    path='your_data_path',
    sensor_list=[0, 1],
    class_list=[0, 1, 2],
    augment=True,
    calibration_config=18
)

# Pose estimation
from dataloaders import pose_dataset
dataset = pose_dataset(
    path='your_data_path',
    sensor_idx=0,
    random_final=False,
    augment=True
)
```

## Network Architectures

The codebase implements several neural network architectures for different tasks:

### Base Model (SITR)
The core architecture is the Sensor-Invariant Tactile Representation (SITR) model, which consists of:
- Vision Transformer backbone with patch embedding
- Calibration-specific components for processing calibration images
- Contrastive learning head for sensor-invariant feature extraction
- Reconstruction decoder for self-supervised learning

### Task-Specific Networks

1. **Classification Network** (`classification_net`)
2. **Pose Estimation Network** (`pose_estimation_net`)
3. **MLP Network** (`MLP_net`)

Example usage:
```python
from models.networks import SITR_base, classification_net, pose_estimation_net, MLP_net

# Initialize base model
base = SITR_base(num_calibration=18)

# Create task-specific networks
classifier = classification_net(base, num_classes=16)
pose_net = pose_estimation_net(base)
mlp_net = MLP_net(base, num_classes=5)
```

## Project Structure

```
.
├── requirements.txt           # Required libraries
├── plot_classifier.py         # Classification evaluation
├── plot_pose_estimater.py     # Pose estimation evaluation
├── dataloaders.py             # Dataset loading utilities
└── models/                    # Neural network models
    ├── networks.py            # Model architectures
    └── losses.py              # Loss functions
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{
    gupta2025sensorinvariant,
    title={Sensor-Invariant Tactile Representation},
    author={Harsh Gupta and Yuchen Mo and Shengmiao Jin and Wenzhen Yuan},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Feel free to reach out at hgupt3-at-illinois.edu if you have any questions.
