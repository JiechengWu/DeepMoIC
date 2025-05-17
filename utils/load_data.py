import pandas as pd
import numpy as np
import random


def load_data(adj, fea, lab, threshold=0.0005):
    print('loading data')
    adj_df = pd.read_csv(adj, header=0, index_col=None)
    fea_df = pd.read_csv(fea, header=0, index_col=None)
    label_df = pd.read_csv(lab, header=None, index_col=None)

    if adj_df.shape[0] != fea_df.shape[0]:
        print('Input files must have same samples.')
        exit(1)

    print('Calculating the laplace adjacency matrix')
    adj_m = adj_df.iloc[:, 1:].values
    # Filter edges with a threshold
    adj_m[adj_m < threshold] = 0

    exist = (adj_m != 0) * 1.0
    # Calculate the degree matrix
    factor = np.ones(adj_m.shape[1])
    res = np.dot(exist, factor)
    diag_matrix = np.diag(res)

    # Calculate the laplace matrix
    d_inv = np.linalg.inv(diag_matrix)
    adj_hat = d_inv.dot(exist)

    return adj_hat, fea_df, label_df.squeeze()



