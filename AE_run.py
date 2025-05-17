import pandas as pd
import numpy as np
import argparse
from model.autoencoder import AE
import torch
import torch.utils.data as Data
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='gpu', help='Training on cpu or gpu, default: gpu.')

    parser.add_argument('--path1', '-p1', default='./data/KIPAN/1.csv', type=str, help='The first omics file name.')
    parser.add_argument('--path2', '-p2', default='./data/KIPAN/2.csv', type=str, help='The second omics file name.')
    parser.add_argument('--path3', '-p3', default='./data/KIPAN/3.csv', type=str, help='The third omics file name.')
    parser.add_argument('--fea_name_path', default=['../data/KIPAN/1_featurename.csv', '../data/KIPAN/2_featurename.csv', '../data/KIPAN/3_featurename.csv'],
                        type=str, nargs=3, help='The third omics file name.')

    parser.add_argument('--batchsize', '-bs', type=int, default=32, help='Training batchszie, default: 64.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('--epoch', type=int, default=100, help='Training epochs, default: 150.')

    parser.add_argument('--a', type=float, default=0.3, help='[0,1], float, weight for the first omics data, 0.6')
    parser.add_argument('--b', type=float, default=0.5, help='[0,1], float, weight for the second omics data, 0.1.')
    parser.add_argument('--c', type=float, default=0.2, help='[0,1], float, weight for the third omics data, 0.3.')

    parser.add_argument('--latent_dim', type=int, default=100, help='Extract features dimensionality, default: 100.')
    parser.add_argument('--dataset', type=str, default="ROSMAP", help='dataset name')
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # get omics data
    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = "./data/"

    args.path1 = data_path + args.dataset + "/1.csv"
    args.path2 = data_path + args.dataset + "/2.csv"
    args.path3 = data_path + args.dataset + "/3.csv"

    if args.dataset == "BRCA":
        omics_data1 = pd.read_csv(args.path1, header=0, index_col=0)
        omics_data2 = pd.read_csv(args.path2, header=0, index_col=0)
        omics_data3 = pd.read_csv(args.path3, header=0, index_col=0)
    else:
        omics_data1 = pd.read_csv(args.path1, header=None, index_col=None)
        omics_data2 = pd.read_csv(args.path2, header=None, index_col=None)
        omics_data3 = pd.read_csv(args.path3, header=None, index_col=None)


    length = len(omics_data1)
    sample_name = list(range(length))
    omics_data1.insert(0, 'Sample', sample_name)
    omics_data2.insert(0, 'Sample', sample_name)
    omics_data3.insert(0, 'Sample', sample_name)

    if args.a + args.b + args.c != 1.0:
        print('The sum of weights must be 1.')
        exit(1)

    # dims of each omics data
    in_feas = [omics_data1.shape[1] - 1, omics_data2.shape[1] - 1, omics_data3.shape[1] - 1]

    data = pd.merge(omics_data1, omics_data2, on='Sample', how='inner')
    data = pd.merge(data, omics_data3, on='Sample', how='inner')
    # data = pd.merge(omics_data1_selected, omics_data2_selected, on='Index', how='inner')
    sample_name = data['Sample'].tolist()

    # change data to a Tensor
    X, Y = data.iloc[:, 1:].values, np.zeros(data.shape[0])
    X_train, Y_train = torch.tensor(X, dtype=torch.float, device=device), torch.tensor(Y, dtype=torch.float, device=device)
    # train model
    print('Begin training')
    Tensor_data = Data.TensorDataset(X_train, Y_train)
    train_loader = Data.DataLoader(Tensor_data, batch_size=args.batchsize, shuffle=True)

    model = AE(in_feas, latent_dim=args.latent_dim, a=args.a, b=args.b, c=args.c)
    model.to(device)
    model.train()
    model.train_AE(train_loader, learning_rate=args.learning_rate, device=device, epochs=args.epoch, dataset=args.dataset)
    model.eval()  # before save and test, fix the variables

    ae_dir = 'result/{}/AE'.format(args.dataset)
    if not os.path.exists(ae_dir):
        os.makedirs(ae_dir)
    torch.save(model, 'result/{}/AE/AE_model.pkl'.format(args.dataset))

    # load saved model
    print('Get the latent layer output...')
    model = torch.load('result/{}/AE/AE_model.pkl'.format(args.dataset))

    omics_1 = X_train[:, :in_feas[0]]
    omics_2 = X_train[:, in_feas[0]:in_feas[0] + in_feas[1]]
    omics_3 = X_train[:, in_feas[0] + in_feas[1]: in_feas[0] + in_feas[1] + in_feas[2]]
    latent_data, _, _, _ = model.forward(omics_1, omics_2, omics_3)
    latent_df = pd.DataFrame(latent_data.detach().cpu().numpy())
    latent_df.insert(0, 'Sample', sample_name)

    latent_df.to_csv('result/{}/latent_data.csv'.format(args.dataset), header=True, index=False)

    print('Finished!')
