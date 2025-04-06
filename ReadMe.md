# All-in-One Medical Image Re-Identification (CVPR2025)

MaMI is the fist unified re-identification model for medical images, capable of handling various imaging modalities such as X-ray, CT, fundus, and pathology images. By leveraging a Continuous Modality-based Parameter Adapter (ComPA) and integrating medical priors, MaMI supports both historical data-assisted diagnosis and privacy protection applications.

<div align="center">
  <img src="main.png" alt="MAMI" style="width:50%; display:block; margin:0 auto;">
</div>



## Features

- **Unified Multi-Modality Model**  
  A single model supports multiple medical image modalities without having to train separate models for each modality.

- **Continuous Modality Adaptive Parameterization**  
  The ComPA module generates continuous modality representations to dynamically adapt model parameters based on the input image.

- **Medical Priors Integration**  
  Incorporates pre-trained Medical Foundation Models (MFMs) to enhance feature discrimination, capturing subtle identity-related cues for more robust re-identification.

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA-enabled GPU (recommended: NVIDIA RTX 4090)

## Pre-trained Model
MAMI_pretrained.pth

Link: https://pan.baidu.com/s/1GD-TsafqYhbXM6VwQVUwIA?pwd=fbge

## Usage

1. You can download the pre-trained model from the above link.

2. copy the file the file "MAMI_pretrained.pth" to this directory.

3. run "test.bash"

## ToDo
- [ ] **Open Source Train/Validation Split**  
  Publish the scripts/configuration used to generate the train/val split for the dataset.

- [ ] **Open Source Training Code**  
  Release the complete training pipeline including all scripts and configuration files.


## Citation
If you find our work useful, please cite:

@article{tian2025towards,
  title={Towards All-in-One Medical Image Re-Identification},
  author={Tian, Yuan and Ji, Kaiyuan and Zhang, Rongzhao and Jiang, Yankai and Li, Chunyi and Wang, Xiaosong and Zhai, Guangtao},
  journal={arXiv preprint arXiv:2503.08173},
  year={2025}
}