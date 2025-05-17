import snf
import pandas as pd
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, nargs=3,
                        help='Location of input files, must be 3 files')
    parser.add_argument('--metric', type=str, choices=['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
                        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'], default='sqeuclidean',
                        help='Distance metric to compute.')
    parser.add_argument('--k', type=int, default=20, help='(0, N) int, number of neighbors to consider when creating affinity matrix. Default: 20.')
    parser.add_argument('--mu', type=int, default=0.5,
                        help='(0, 1) float, Normalization factor to scale similarity kernel when constructing affinity matrix. Default: 0.5.')

    parser.add_argument('--dataset', default='ROSMAP', type=str, help='.')

    args = parser.parse_args()

    data_path = "./data/"

    args.path = [data_path + "{}/1.csv".format(args.dataset),
                 data_path + "{}/2.csv".format(args.dataset),
                 data_path + "{}/3.csv".format(args.dataset)]

    if args.dataset == "BRCA":
        omics_data_1 = pd.read_csv(args.path[0], header=0, index_col=0)
        omics_data_2 = pd.read_csv(args.path[1], header=0, index_col=0)
        omics_data_3 = pd.read_csv(args.path[2], header=0, index_col=0)
    else:
        omics_data_1 = pd.read_csv(args.path[0], header=None, index_col=None)
        omics_data_2 = pd.read_csv(args.path[1], header=None, index_col=None)
        omics_data_3 = pd.read_csv(args.path[2], header=None, index_col=None)

    length = len(omics_data_1)
    sample_name = list(range(length))
    omics_data_1.insert(0, 'Sample', sample_name)
    omics_data_2.insert(0, 'Sample', sample_name)
    omics_data_3.insert(0, 'Sample', sample_name)

    if omics_data_1.shape[0] != omics_data_2.shape[0] or omics_data_1.shape[0] != omics_data_3.shape[0]:
        print('Input files must have same samples.')
        exit(1)

    print('Start fusion')
    affinity_nets = snf.make_affinity(
        [omics_data_1.iloc[:, 1:].values.astype(np.float64), omics_data_2.iloc[:, 1:].values.astype(np.float64), omics_data_3.iloc[:, 1:].values.astype(np.float64)],
        metric=args.metric, K=args.k, mu=args.mu)

    fused_net = snf.snf(affinity_nets, K=args.k)

    fused_df = pd.DataFrame(fused_net)
    fused_df.columns = omics_data_1['Sample'].tolist()
    fused_df.index = omics_data_1['Sample'].tolist()
    fused_df.to_csv('result/{}/PSN_matrix.csv'.format(args.dataset), header=True, index=True)

    print('Finished!')