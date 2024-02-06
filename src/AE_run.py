import pandas as pd
import numpy as np
import argparse
from model.autoencoder import AE
import torch
import torch.utils.data as Data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('--path1', '-p1', default='data/TCGA/common_expression_data.csv', type=str, help='The first omics file path.')
    parser.add_argument('--path2', '-p2', default='data/TCGA/common_cnv_data.csv', type=str, help='The second omics file path.')
    parser.add_argument('--batchsize', '-bs', type=int, default=64, help='Training batchszie, default: 64.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('--epoch', '-e', type=int, default=150, help='Training epochs, default: 150.')
    parser.add_argument('--latent', '-l', type=int, default=1000, help='The latent layer dim, default: 1000.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='gpu', help='Training on cpu or gpu, default: gpu.')
    parser.add_argument('--a', '-a', type=float, default=0.5, help='[0,1], float, weight for the first omics data')
    parser.add_argument('--b', '-b', type=float, default=0.5, help='[0,1], float, weight for the second omics data.')
    parser.add_argument('--latent_dim', '-n', type=int, default=100, help='Extract top N features every 10 epochs, default: 100.')
    parser.add_argument('--dataset', type=str, default="TCGA", help='dataset name')
    args = parser.parse_args()

    # get omics data
    omics_data1 = pd.read_csv(args.path1, sep='\t', header=0, index_col=None)
    omics_data2 = pd.read_csv(args.path2, sep='\t', header=0, index_col=None)

    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.a + args.b != 1.0:
        print('The sum of weights must be 1.')
        exit(1)

    # dims of each omics data
    in_feas = [omics_data1.shape[1] - 3, omics_data2.shape[1] - 3]

    omics_data1.rename(columns={omics_data1.columns.tolist()[0]: 'Index'}, inplace=True)
    omics_data2.rename(columns={omics_data2.columns.tolist()[0]: 'Index'}, inplace=True)
    omics_data1_selected = omics_data1.iloc[:, :-2]
    omics_data2_selected = omics_data2.iloc[:, :-2]

    # merge the multi-omics data
    data = pd.merge(omics_data1_selected, omics_data2_selected, on='Index', how='inner')
    sample_name = data['Index'].tolist()

    # change data to a Tensor
    X, Y = data.iloc[:, 1:].values, np.zeros(data.shape[0])
    X_train, Y_train = torch.tensor(X, dtype=torch.float, device=device), torch.tensor(Y, dtype=torch.float, device=device)
    # train a AE model
    print('Training model')
    Tensor_data = Data.TensorDataset(X_train, Y_train)
    train_loader = Data.DataLoader(Tensor_data, batch_size=args.batchsize, shuffle=True)

    model = AE(in_feas, latent_dim=args.latent_dim, a=args.a, b=args.b)
    model.to(device)
    model.train()
    model.train_AE(train_loader, learning_rate=args.learning_rate, device=device, epochs=args.epoch, dataset=args.dataset)
    model.eval()  # before save and test, fix the variables
    torch.save(model, '../result/{}/AE/AE_model.pkl'.format(args.dataset))

    # load saved model
    print('Get the latent layer output...')
    model = torch.load('../result/{}/AE/AE_model.pkl'.format(args.dataset))

    omics_1 = X_train[:, :in_feas[0]]
    omics_2 = X_train[:, in_feas[0]:in_feas[0] + in_feas[1]]
    latent_data = model.forward(omics_1, omics_2)
    latent_df = pd.DataFrame(latent_data.detach().cpu().numpy())
    latent_df.insert(0, 'Index', sample_name)

    latent_df.to_csv('../result/{}/latent_data.csv'.format(args.dataset), header=True, index=False)

    print('Finished!')
