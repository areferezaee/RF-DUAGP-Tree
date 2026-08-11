#RF-DUAGP-Tree

This repository contains the implementation of **DUAGP-Tree**, a dynamics- and uncertainty-aware Gaussian Process Tree for surgical skill assessment.

## Overview

DUAGP-Tree is designed for surgical skill assessment using video-based representations and uncertainty-aware Gaussian Process Tree modeling.

The implementation includes support for the **JIGSAWS** and **Capsulorhexis** datasets and provides components for feature processing, dataset loading, Gaussian Process modeling, uncertainty-aware kernels, and tree-based learning.

## Repository Structure

```text
DUAGP-Tree/
├── datasets/
│   ├── capsulorhexis.py
│   ├── capsulorhexis_loader.py
│   ├── JIGSAWS.py
│   └── jigsaws_loader.py
│
├── GP_Tree/
│   ├── class_splits.py
│   ├── gp_model.py
│   ├── gp_model_gibbs.py
│   ├── JK.py
│   ├── kernel_class.py
│   ├── Learner.py
│   ├── node.py
│   └── tree.py
│
├── trainer.py
├── utils.py
└── darawconfusioonm.py
```

## Datasets

The code provides dataset interfaces for:

* **JIGSAWS**
* **Capsulorhexis**

The dataset files are not included in this repository. Please obtain the datasets from their respective sources and configure the dataset paths in the code before running the experiments.
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
git clone https://github.com/areferezaee/DUAGP-Tree.git
cd DUAGP-Tree
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
python3 trainer.py --dataset='JIGSAWS' --natural-lr=.9 --num-inducing-points=15 --outputscale=1.
```

### Capsulorhexis

To run DUAGP-Tree on the Capsulorhexis dataset:

```bash
python3 trainer.py --dataset='Capsulorhexis' --natural-lr=.9 --num-inducing-points=8 --outputscale=1.
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

The implementations of `GP_Tree/node.py`, `GP_Tree/tree.py`, and `GP_Tree/Learner.py` are derived from the original GP-Tree implementation and incorporated into the DUAGP-Tree framework.

The kernel implementations in `GP_Tree/JK.py` and `GP_Tree/kernel_class.py` were developed specifically for this work. `JK.py` is named separately to distinguish the two kernel implementations used in this repository.

The implementation in `GP_Tree/gp_model.py` has also been substantially modified and extended for the proposed DUAGP-Tree framework.

We gratefully acknowledge the authors of the original GP-Tree implementation.

**Original GP-Tree project:**
https://github.com/yannickach/GP-Tree

### Reference

[1] I. Achituve, A. Navon, Y. Yemini, G. Chechik, and E. Fetaya, "GP-Tree: A Gaussian Process Classifier for Few-Shot Incremental Learning," *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 2021.
