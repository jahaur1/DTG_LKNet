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

### Cite
If you find the paper useful, please cite as following:


Thanks to the following open-source repositories for their valuable support in this work:

- [LCDFormer](https://github.com/NanakiC/LCDFormer)
- [ASTGNN](https://github.com/guoshnBJTU/ASTGNN)
- [ConvTimeNet](https://github.com/Mingyue-Cheng/ConvTimeNet)
- [PDFormer](https://github.com/BUAABIGSCity/PDFormer)
- [RepLKNet-pytorch](https://github.com/DingXiaoH/RepLKNet-pytorch)

