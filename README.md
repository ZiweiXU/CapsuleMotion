# Introduction
This repository contains the software and T20 dataset used in NeurIPS 2021 paper 
[Unsupervised Motion Representation Learning with Capsule Autoencoders](https://arxiv.org/abs/2110.00529).

# Prerequisites

- Recommended environment: Ubuntu 18.04 with CUDA 11.2 and CuDNN 7.6.5.
- Required Python packages: listed in `requirements.txt`.

## Dataset

T20 and NW-UCLA are included in the repository.

For NTURGBD-60/120, download `nturgbd32.tar.gz` [here](https://drive.google.com/drive/folders/1pxnsW3ocn_6PngB134crznFAzV1Gw2rq?usp=sharing).
Place it in `data/`, then extract it with `tar -xzf nturgbd32.tar.gz`.
This should produce `data/nturgbd32`.

## Pretrained Models

Download `checkpoints.tar.gz` [here](https://drive.google.com/drive/folders/1pxnsW3ocn_6PngB134crznFAzV1Gw2rq?usp=sharing).
Place the file at project root, extract it and checkpoints will be released to `checkpoints/${DATASET_SUBDIR}/`.

# Experiments

## Basic Usage

```bash
python main.py @${CONFIG_FILE}
```

To run a pre-defined full experiment, replace `${CONFIG_FILE}` with one of the 
`txt` files in `configs/`. For example, `python main.py @configs/t20.txt`.

To test a pretrained model, replace `${CONFIG_FILE}` with the config in 
`checkpoints/${DATASET_SUBDIR}`. For example,
`python main.py @checkpoints/nturgbd/60_xsub/config.txt`.

## About Config Files

A config file is essentially a line-separated list of command line parameters 
passed to `main.py`.
If you feel like editing configs for fun, please note:

1. `--config_path` must point to the path of the config file relative to project
root. 
2. `--model_params`, `--ds_params`, `--lrsch_params` and `--loss_weights` are 
json strings of keyword parameters used to initialize model/dataset/lrsch or 
calculating loss. For `--model_params`, see `lib/mcae/mp.py:53`. For `--loss_weights`, 
see `lib/mcae/mp.py:169`. 
3. Type `python main.py --help` for other details.

# Credits

Some math/spatial operations are adapted from 
[SCAE](https://github.com/akosiorek/stacked_capsule_autoencoders) and 
[DDPAE](https://github.com/jthsieh/DDPAE-video-prediction).
Code from [MS-G3D](https://github.com/kenziyuliu/MS-G3D) are used to
preprocess NTURGBD60/120.
We would like to thank the authors for their contribution.

# Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@misc{xu2021unsupervised,
      title={Unsupervised Motion Representation Learning with Capsule Autoencoders}, 
      author={Ziwei Xu and Xudong Shen and Yongkang Wong and Mohan S Kankanhalli},
      year={2021},
      eprint={2110.00529},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
