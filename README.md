# DUAGP-Tree

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

The main training script is:

```bash
python trainer.py
```

Dataset-specific processing and loading are implemented in the `datasets/` directory.

## Evaluation

Evaluation and visualization utilities are provided in:

```text
utils.py
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
