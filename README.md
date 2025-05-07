# DPFFN: A Dual-Polarization Feature Fusion Network for Radar Automatic Target Recognition

This repository provides the official PyTorch implementation of **DPFFN**, a novel deep learning framework designed for radar automatic target recognition (RATR) using High-Resolution Range Profile (HRRP) sequences under dual-polarization settings. DPFFN extracts and fuses global-local features across HH and VH channels to improve classification performance.

## 🔍 Overview

DPFFN introduces a two-branch architecture to exploit both **global structural semantics** and **local scattering details**. Key innovations include:

- **Dual-Polarization Tokenization**: Separately processes HH and VH HRRP signals into feature tokens.
- **Global and Local Subnetworks**: Captures complementary characteristics at different semantic levels.
- **Two-Stage Feature Fusion**: Integrates features hierarchically across polarizations and semantic scales.
- **Fusion Loss**: Enhances consistency and discriminability across modalities.

<!-- If available, include a network diagram here -->
<!-- 
<p align="center">
  <img src="assets/framework.png" alt="DPFFN Framework" width="80%">
</p> 
-->

## 📁 Repository Structure

```
DPFFN/
│
├── model/                  # DPFFN model architecture
│   └── dpffn.py
├── data/                   # Dataset reading and preprocessing tools
│   └── data_loader.py
├── loss/                   # Custom loss functions (e.g., fusion loss)
│   └── loss_fn.py
├── utils/                  # Miscellaneous utilities
├── train.py                # Training script
├── test.py                 # Evaluation and confusion matrix
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Environment Setup

```bash
git clone https://github.com/xmpan/DPFFN.git
cd DPFFN
pip install -r requirements.txt
```

### Dataset

Please prepare your HRRP dataset in the format `[Batch, Sequence Length, Channel]`, e.g., `[N, 512, 401]` for HH/VH channels. Update the path and format loader in `data/data_loader.py`.

> ⚠️ We do not distribute radar data due to security and copyright restrictions.

### Training

```bash
python train.py
```

### Evaluation

```bash
python test.py
```

This will evaluate the model and generate confusion matrices and performance metrics.

## 📊 Results

| Method    | Accuracy (%) | Params (M) | Polarization | Notes                |
|-----------|--------------|------------|--------------|----------------------|
| DPFFN     | 94.7         | 4.8        | HH + VH      | With fusion loss     |
| Baseline  | 89.3         | 4.5        | HH only      | No fusion strategy   |

## 📄 Citation

If you find this work helpful in your research, please consider citing:

```bibtex
@article{zhou2025dpffn,
  title={A Dual-Polarization Feature Fusion Network for Radar Automatic Target Recognition Based on HRRP Sequences},
  author={Zhou, Yangbo and others},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  year={2025},
  note={Submitted}
}
```

## 📬 Contact

For questions or collaborations, please contact:  
[yangbo_zhou2001@163.com](mailto:yangbo_zhou2001@163.com)

---

## 🔒 License

This project is licensed under the MIT License.
