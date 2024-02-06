import numpy as np
import argparse
import glob
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import torch
import torch.nn.functional as F
from utils.load_data import load_data
from model.deepGCN import DeepGCN


def train(epoch, optimizer, features, adj, labels, idx_train):
    labels.to(device)

    DeepGCN_model.train()
    optimizer.zero_grad()
    output = DeepGCN_model(features, adj)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])

    ot = output[idx_train].detach().cpu().numpy()
    ot = np.argmax(ot, axis=1)
    lb = labels[idx_train].detach().cpu().numpy()
    acc_train = accuracy_score(lb, ot)

    loss_train.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print('Epoch: %.2f, Loss_train: %.4f, Acc_train: %.4f' % (epoch + 1, loss_train.item(), acc_train.item()))
    return acc_train.item(), loss_train.data.item()


def test(features, adj, labels, idx_test):
    DeepGCN_model.eval()
    output = DeepGCN_model(features, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])

    ot = output[idx_test].detach().cpu().numpy()
    ot = np.argmax(ot, axis=1)
    lb = labels[idx_test].detach().cpu().numpy()

    acc = accuracy_score(lb, ot)
    f = f1_score(lb, ot, average='macro')
    p = precision_score(lb, ot, average="macro")
    r = recall_score(lb, ot, average="macro")

    print("Test results:\n",
          "loss = {:.4f}\n".format(loss_test.item()),
          "acc = {:.4f}\n".format(acc.item()),
          "maF1 = {:.4f}\n".format(f.item()),
          "maP = {:.4f}\n".format(p.item()),
          "maR = {:.4f}\n".format(r.item()))

    return acc, f, p, r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--featuredata', '-fd', type=str, default="result/TCGA/latent_data.csv", help='The feature file.')
    parser.add_argument('--adjdata', '-ad', type=str, default="result/TCGA/PSN_matrix.csv", help='The adjacency matrix file.')
    parser.add_argument('--labeldata', '-ld', type=str, default="data/TCGA/sample_classes.csv", help='The sample label file.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='gpu',
                        help='Training on cpu or gpu, default: cpu.')
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Training epochs, default: 300.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('--weight_decay', '-w', type=float, default=0.01,
                        help='Weight decay (L2 loss on parameters), methods to avoid overfitting, default: 0.01')
    parser.add_argument('--dropout', '-dp', type=float, default=0.3,
                        help='Dropout rate, methods to avoid overfitting, default: 0.3.')
    parser.add_argument('--threshold', '-t', type=float, default=0.0005,
                        help='Threshold to filter edges, default: 0.0005')
    parser.add_argument('--nclass', '-nc', type=int, default=28, help='Number of classes, default: 28')
    parser.add_argument('--patience', '-p', type=int, default=20, help='Patience')
    parser.add_argument('--n_repeat', type=int, default=5, help='repeat times')

    parser.add_argument('--dataset', type=str, default="TCGA", help='datastet name')
    parser.add_argument('--hidden', '-hd', type=int, default=512, help='Hidden layer dimension, default: 512')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha for ini')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda')

    parser.add_argument('--ratio', type=float, default=0.6, help='train size ratio')

    args = parser.parse_args()

    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load input files
    adj, data, label = load_data(args.adjdata, args.featuredata, args.labeldata, args.threshold)

    adj = torch.tensor(adj, dtype=torch.float, device=device)
    features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float, device=device)
    labels = torch.tensor(label.iloc[:, 1].values, dtype=torch.long, device=device)

    print('Begin training model...')
    print("ratio: {}".format(args.ratio))
    # save sample name
    all_sample = data['Index'].tolist()

    # shuffle sample index
    shuffle_index = np.random.permutation(len(labels))

    train_size = int(len(shuffle_index) * args.ratio)
    train_idx = shuffle_index[0:train_size]
    test_idx = shuffle_index[train_size:]

    all_ACC = []
    all_F1 = []
    all_P = []
    all_R = []

    for i in range(args.n_repeat):
        DeepGCN_model = DeepGCN(nfeat=features.shape[1], nclass=args.nclass, dim=args.hidden,
                            alpha=args.alpha, lamda=args.lamda, dropout=args.dropout)
        DeepGCN_model.to(device)
        optimizer = torch.optim.Adam(DeepGCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)

        idx_train, idx_test = torch.tensor(train_idx, dtype=torch.long, device=device), torch.tensor(test_idx, dtype=torch.long, device=device)

        loss_values = []  
        acc_values = []
        # record the times with no loss decrease, record the best epoch
        bad_counter, best_epoch = 0, 0
        best = 1000  # record the lowest loss value
        for epoch in range(args.epochs):
            acc_train, loss_train = train(epoch, optimizer, features, adj, labels, idx_train)
            loss_values.append(loss_train)
            acc_values.append(acc_train)

            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1 

            if bad_counter == args.patience:
                break

            # save model of this epoch
            torch.save(DeepGCN_model.state_dict(), '../result/{}/model/{}.pkl'.format(args.dataset, epoch))

            # reserve the best model, delete other models
            files = glob.glob('../result/{}/model/*.pkl'.format(args.dataset))
            for file in files:
                name = file.split('\\')[1]
                epoch_nb = int(name.split('.')[0])
                if epoch_nb != best_epoch:
                    os.remove(file)

        print('Training finished.')
        print('The best epoch model is ', best_epoch)
        DeepGCN_model.load_state_dict(torch.load('../result/{}/model/{}.pkl'.format(args.dataset, best_epoch)))
        test_acc, f1, p, r = test(features, adj, labels, idx_test)

        all_ACC.append(test_acc)
        all_F1.append(f1)
        all_P.append(p)
        all_R.append(r)

    fp = open("{}_results.txt".format(args.dataset), "a+", encoding="utf-8")
    fp.write("ratio: {}\n".format(args.ratio))
    fp.write("ACC: {:.2f}({:.2f})\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    fp.write("MaF: {:.2f}({:.2f})\n".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
    fp.write("MaP: {:.2f}({:.2f})\n".format(np.mean(all_P) * 100, np.std(all_P) * 100))
    fp.write("MaR: {:.2f}({:.2f})\n\n".format(np.mean(all_R) * 100, np.std(all_R) * 100))
    fp.close()

    print('Finished!')
