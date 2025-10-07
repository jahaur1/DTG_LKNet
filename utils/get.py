import numpy as np
import torch
from scipy import sparse as sp
import argparse
def get_lap_pos_enc(A,d_model):
    '''

    :param adj_mx:  (N,N)
    :num_of_vertices: N
    :return:
        '''
    num_nodes=A.shape[0]
    N = np.diag(1.0/np.sum(A, axis=1))
#     L = np.dot(np.dot(N, A),N)
    L=sp.eye(num_nodes)-np.dot(np.dot(N, A),N)
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    lap_pos_enc=EigVec[:,1:d_model+1]
    return lap_pos_enc 

def get_mx_mask(adj_mx,num_of_vertices):
    '''

    :param adj_mx:  (N,N)
    :num_of_vertices: N
    :return:
        '''
    adj_mx[adj_mx > 0] = 1
    adj_mx[adj_mx == 0] = 511
    for i in range(num_of_vertices):
        adj_mx[i, i] = 0
    for k in range(num_of_vertices):
        for i in range(num_of_vertices):
            for j in range(num_of_vertices):
                adj_mx[i, j] = min(adj_mx[i, j], adj_mx[i, k] + adj_mx[k, j], 511)
    adj_mx = adj_mx.T
    mask_mx = torch.zeros(num_of_vertices, num_of_vertices)
    mask_mx[adj_mx >= 5] = 1
    mask_mx = mask_mx.bool()
    return mask_mx  

def get_dtw_mask(adj_dtw,num_of_vertices):

    mean = np.mean(adj_dtw)
    std = np.std(adj_dtw)
    adj_dtw = (adj_dtw - mean) / std
    adj_dtw = np.exp(-adj_dtw ** 2 / (0.1) ** 2)
    mask_dtw = torch.zeros(num_of_vertices, num_of_vertices)
    mask_dtw[adj_dtw > 0.6] = 1
    mask_dtw1 = mask_dtw.bool()
    return mask_dtw1
'''
def get_dtw_mask(adj_dtw, num_of_vertices):

    # 计算均值和标准差
    mean = np.mean(adj_dtw)
    std = np.std(adj_dtw)

    # 标准化处理
    adj_dtw_normalized = (adj_dtw - mean) / std

    # 应用高斯核函数
    adj_dtw_gaussian = np.exp(-adj_dtw_normalized ** 2 / (0.1 ** 2))

    # 将numpy数组转换为torch张量
    mask_dtw = torch.tensor(adj_dtw_gaussian, dtype=torch.float32)

    return mask_dtw
'''