[![Paper](https://img.shields.io/badge/Paper-PeerJ%20Computer%20Science-blue)](https://peerj.com/articles/cs-3793/)
[![DOI](https://img.shields.io/badge/DOI-10.7717%2Fpeerj--cs.3793-green)](https://doi.org/10.7717/peerj-cs.3793)
[![Python 3.9](https://img.shields.io/badge/python-3.9.2-blue.svg)](https://www.python.org/downloads/release/python-392/)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.13+-orange)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

# DTG-LKNet: dual spatio-temporal graphs and large-kernel convolutions network for traffic prediction

## Requirements

python.

torch-gpu.

## Data Preparation

Step1: Download datasets([PEMS03](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS03),[PEMS04](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS04),[PEMS07](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS07)).

Step2: Process raw data

```bash
python PrepareData.py
```

Step3: Generate DTW data

```bash
python create_dtw.py
```

## Train

```bash
python run.py
```

### Config

You can modify the parameters in the [configurations](/configurations/).

### Attention

When using PEMS07, please ensure that you have approximately 40GB of GPU memory.

If unable to run PrepareData.py, you can modify your virtual memory based on the error message.
### ERF Visualization for Convolution Layers
![ERF‑TCN comparison](./figure/erf_tcn.png "Large‑Kernel Conv vs standard TCN ERF")
> Figure A: Effective receptive field comparison. Top: large‑kernel convolution; Bottom: standard TCN. Darker color represents higher contribution weight for prediction.

![ERF additional visualization](./figure/erf.png "Supplementary ERF heatmap")
> Figure B: Supplementary effective‑receptive‑field heatmap for traffic prediction.

erf_conv.py is used to calculate and visualize the Effective Receptive Field (ERF) of convolutions. It quantifies the sensitivity of the target layer of the model to input spatiotemporal data (nodes × time steps) through gradient backpropagation, fuses ERF results from multiple test samples to reduce single-sample noise, and focuses on visualizing the ERF distribution of the central node. It intuitively demonstrates the model's attention patterns to input information from different time steps and nodes when predicting the traffic flow of the central node, helping to understand the model's dependence on input spatiotemporal features in traffic flow prediction tasks. The module captures the output of the target layer by registering a forward hook, calculates the input gradient through backpropagation using the mean value of the central features of the target layer as the loss, optimizes the visualization effect through inverse normalization and logarithmic scaling, and finally generates and saves the ERF heatmap of the central node. The number of samples, target layer, or the option to view the ERF distribution of all nodes can be adjusted as needed.
### Cite
If you find the paper useful, please cite as following:

```bibtex
@article{cao2026dtg,
  title={DTG-LKNet: dual spatio-temporal graphs and large-kernel convolutions network for traffic prediction},
  author={Cao, Jiahao and Tian, Yuan and Long, YangSheng and Wang, Peng and Xiao, Tong and Ye, Peng and Teng, Guoqing},
  journal={PeerJ Computer Science},
  volume={12},
  pages={e3793},
  year={2026},
  publisher={PeerJ Inc.}
}
```
Thanks to the following open-source repositories for their valuable support in this work:

- [LCDFormer](https://github.com/NanakiC/LCDFormer)
- [ASTGNN](https://github.com/guoshnBJTU/ASTGNN)
- [ConvTimeNet](https://github.com/Mingyue-Cheng/ConvTimeNet)
- [PDFormer](https://github.com/BUAABIGSCity/PDFormer)
- [RepLKNet-pytorch](https://github.com/DingXiaoH/RepLKNet-pytorch)

