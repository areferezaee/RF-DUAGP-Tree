## RF-DUAGP-Tree: Video-based Surgical Skill Assessment Using Dynamics-and-Uncertainty-Aware Tree-based Gaussian Process Classifier

This repository contains the implementation of **RF-DUAGP-Tree**, a Representation Flow-based Dynamics and Uncertainty-Aware Gaussian Process Tree for surgical skill assessment.

## Overview

RF-DUAGP-Tree is designed for surgical skill assessment using video-based representations and uncertainty-aware Gaussian Process Tree modeling.

The implementation includes support for the **JIGSAWS** and **Capsulorhexis** datasets and provides components for feature processing, dataset loading, Gaussian Process modeling, uncertainty-aware kernels, and tree-based learning.

## Repository Structure

```text
RF-DUAGP-Tree/
├── feature_engineering/
│   ├── capsulorhexis.py
│   ├── capsulorhexis_loader.py
│   ├── JIGSAWS.py
│   └── jigsaws_loader.py
│
├── DUAGP_Tree/
│   ├── class_splits.py
│   ├── duagp_model.py
│   ├── gp_model_gibbs.py
│   ├── jig_kernel.py
│   ├── caps_kernel.py
│   ├── dualearner.py
│   ├── duanode.py
│   └── duatree.py
│
├── main.py
├── utils.py
└── darawconfusioonm.py
```

## Datasets and Feature Engineering

The code provides interfaces for the following datasets:

```text
JIGSAWS
Capsulorhexis
```

Dataset files are not included in this repository. Please obtain the datasets from their respective sources and configure the corresponding dataset paths in the code before running the experiments.

## Feature Extraction

Feature representations are extracted using a **Representation Flow CNN** before being processed by the DUAGP-Tree framework.

For the JIGSAWS dataset, the extracted representations are further processed to construct temporal dynamics, including feature, velocity, acceleration, and jerk representations.

For the Capsulorhexis dataset, the extracted representations are summarized using statistical and information-theoretic descriptors before being provided to the DUAGP-Tree model.

The Representation Flow component is based on the approach introduced by Piergiovanni and Ryoo [2].

### Reference

[2] A. J. Piergiovanni and M. S. Ryoo, "Representation Flow for Action Recognition," *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 9945–9953.

## Method Components

The repository includes:

* Gaussian Process Tree modeling
* Dynamics-aware feature processing
* Uncertainty-aware kernels
* Temporal feature representations
* JIGSAWS and Capsulorhexis data loaders
* Training and evaluation utilities

The implementation includes the uncertainty-aware kernel variants **UA1**, **UA2**, and **EA**.

## Installation

Clone the repository:

```bash
git clone https://github.com/areferezaee/RF-DUAGP-Tree.git
cd RF-DUAGP-Tree
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

> The `requirements.txt` file will be provided with the repository.

## Usage

### JIGSAWS

To run DUAGP-Tree on the JIGSAWS dataset:

```bash
python3 main.py --dataset='JIGSAWS' --natural-lr=.9 --num-inducing-points=15 --outputscale=1.
```

### Capsulorhexis

To run DUAGP-Tree on the Capsulorhexis dataset:

```bash
python3 main.py --dataset='Capsulorhexis' --natural-lr=.9 --num-inducing-points=8 --outputscale=1.
```


Dataset-specific processing and loading are implemented in the `datasets/` directory.

## Evaluation

Evaluation and visualization utilities are provided in:

```text
darawconfusioonm.py
```

The experimental configuration and dataset paths should be adapted to the local environment before running the code.

## Results

Experimental results will be reported here as the corresponding experiments and paper results are finalized.

## Citation

If you use this repository in your research, please cite the corresponding paper:

> Citation information will be added after publication.

## License

A license will be added to this repository.

## Acknowledgement and Code Attribution

This repository builds upon the GP-Tree framework by Achituve et al. [1].

The implementations of `DUAGP_Tree/duanode.py`, `DUAGP_Tree/duatree.py`, and `DUAGP_Tree/dualearner.py` are derived from the original GP-Tree implementation and substantially modified and extended to incorporate noise-robust components and NIS propagation within the proposed RF-DUAGP-Tree framework.

The kernel implementations in `DUAGP_Tree/jig_kernel.py` and `DUAGP_Tree/caps_kernel.py` were independently developed for this work and constitute part of the proposed RF-DUAGP-Tree framework.

The implementation originally provided in `DUAGP_Tree/gp_model.py` was substantially modified and extended as `DUAGP_Tree/duagp_model.py` for the proposed RF-DUAGP-Tree framework.

We gratefully acknowledge the authors of the original GP-Tree implementation.

**Original GP-Tree project:**
https://github.com/IdanAchituve/GP-Tree

### Reference

[1] I. Achituve, A. Navon, Y. Yemini, G. Chechik, and E. Fetaya, "GP-Tree: A Gaussian Process Classifier for Few-Shot Incremental Learning," *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 2021.

