import snf
import pandas as pd
import numpy as np
import argparse
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', '-p1', default='data/TCGA/common_expression_data.csv', type=str, help='The first omics file name.')
    parser.add_argument('--path2', '-p2', default='TCGA/common_cnv_data.csv', type=str, help='The second omics file name.')
    parser.add_argument('--dataset', type=str, default="TCGA", help='dataset name')
    parser.add_argument('--metric', '-m', type=str, choices=['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
                        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'], default='sqeuclidean',
                        help='Distance metric to compute.')
    parser.add_argument('--K', '-k', type=int, default=20,
                        help='(0, N) int, number of neighbors to consider when creating affinity matrix. Default: 20.')
    parser.add_argument('--mu', '-mu', type=int, default=0.5,
                        help='(0, 1) float, Normalization factor to scale similarity kernel when constructing affinity matrix. Default: 0.5.')
    args = parser.parse_args()

    print('Load data files')
    omics_data_1 = pd.read_csv(args.path0, sep='\t', header=0, index_col=None)
    omics_data_2 = pd.read_csv(args.path1, sep='\t', header=0, index_col=None)
    omics_data_1 = omics_data_1.iloc[:, :-2]
    omics_data_2 = omics_data_2.iloc[:, :-2]
    print(omics_data_1.shape, omics_data_2.shape)

    if omics_data_1.shape[0] != omics_data_2.shape[0]:
        print('Input files must have same samples.')
        exit(1)

    omics_data_1.rename(columns={omics_data_1.columns.tolist()[0]: 'Index'}, inplace=True)
    omics_data_2.rename(columns={omics_data_2.columns.tolist()[0]: 'Index'}, inplace=True)

    print('Start fusion')
    affinity_nets = snf.make_affinity(
        [omics_data_1.iloc[:, 1:].values.astype(np.float64), omics_data_2.iloc[:, 1:].values.astype(np.float64)],
        metric=args.metric, K=args.K, mu=args.mu)

    fused_net = snf.snf(affinity_nets, K=args.K)

    fused_df = pd.DataFrame(fused_net)
    fused_df.columns = omics_data_1['Index'].tolist()
    fused_df.index = omics_data_1['Index'].tolist()
    fused_df.to_csv('../result/{}/PSN_matrix.csv'.format(args.dataset), header=True, index=True)

    print('Finished!')