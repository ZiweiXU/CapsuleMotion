# Introduction
This repository contains the software and T20 dataset used in NeurIPS 2021 paper 
"Unsupervised Motion Representation Learning with Capsule Autoencoders".

# Prerequisites

Ubuntu 18.04 with CUDA 11.2 and CuDNN 7.6.5 is recommended.

Required Python modules are listed in `requirements.txt`.

T20 and NW-UCLA experiments are directly runnable (see below).
Data and config for NTURGBD-60/120 will be provided soon.

# Experiments

The most straightforward way to run a full experiment is to run the following at
the project root:

```bash
python main.py @${CONFIG_FILE}
```
Replace `${CONFIG_FILE}` with one of the `txt` files in `configs/`. For example

```bash
python main.py @configs/t20.txt
```

## Test a Checkpoint

Trained models are provided in `checkpoints/${DATASET_SUBDIR}/`.
Each directory contains a checkpoint and its test config.
To load and test a checkpoint, replace `${CONFIG_FILE}` with the config in 
the corresponding `${DATASET_SUBDIR}`. For example

```bash
python main.py @checkpoints/t20/config.txt
```

## About Config Files

A config file is essentially a line-separated list of command line parameters 
passed to `main.py`. Please see 
[this stackoverflow answer](https://stackoverflow.com/a/48651121).

If you feel like editing configs for fun, please note:

1. `--config_path` must point to the path of the config file relative to project
root. 
2. `--model_params`, `--ds_params`, `--lrsch_params` and `--loss_weights` are 
json strings. 

# Credits

Some math/spatial operations are adapted from 
[SCAE](https://github.com/akosiorek/stacked_capsule_autoencoders) and 
[DDPAE](https://github.com/jthsieh/DDPAE-video-prediction).

# Citation

If this repository is useful in your research, please cite 
[Unsupervised Motion Representation Learning with Capsule Autoencoders]()
(Bib entry coming soon).
