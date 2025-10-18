# DTG_LKNet

## Requirements

python.

torch-gpu.

## Data Preparation

Step1: Download datasets([PEMS03](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS03),[PEMS04](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS04),[PEMS07](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS07).

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
erf_conv.py is used to calculate and visualize the Effective Receptive Field (ERF) of convolutions. It quantifies the sensitivity of the target layer of the model to input spatiotemporal data (nodes × time steps) through gradient backpropagation, fuses ERF results from multiple test samples to reduce single-sample noise, and focuses on visualizing the ERF distribution of the central node. It intuitively demonstrates the model's attention patterns to input information from different time steps and nodes when predicting the traffic flow of the central node, helping to understand the model's dependence on input spatiotemporal features in traffic flow prediction tasks. The module captures the output of the target layer by registering a forward hook, calculates the input gradient through backpropagation using the mean value of the central features of the target layer as the loss, optimizes the visualization effect through inverse normalization and logarithmic scaling, and finally generates and saves the ERF heatmap of the central node. The number of samples, target layer, or the option to view the ERF distribution of all nodes can be adjusted as needed.
### Cite
If you find the paper useful, please cite as following:


Thanks to the following open-source repositories for their valuable support in this work:

- [LCDFormer](https://github.com/NanakiC/LCDFormer)
- [ASTGNN](https://github.com/guoshnBJTU/ASTGNN)
- [ConvTimeNet](https://github.com/Mingyue-Cheng/ConvTimeNet)
- [PDFormer](https://github.com/BUAABIGSCity/PDFormer)
- [RepLKNet-pytorch](https://github.com/DingXiaoH/RepLKNet-pytorch)

