import numpy as np
import argparse
import glob
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import torch
import torch.nn.functional as F
from utils.load_data import load_data
from model.deepGCN import DeepGCN
from tqdm import tqdm
from copy import deepcopy


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
    # if (epoch + 1) % 10 == 0:
    #     print('Epoch: %.2f, Loss_train: %.4f, Acc_train: %.4f' % (epoch + 1, loss_train.item(), acc_train.item()))
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

    parser.add_argument('--seed', '-s', type=int, default=1024, help='Random seed, default=0.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='gpu',
                        help='Training on cpu or gpu, default: cpu.')

    parser.add_argument('--epochs', '-e', type=int, default=300, help='Training epochs, default: 300.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.001,
                        help='Weight decay (L2 loss on parameters), methods to avoid overfitting, default: 0.01')
    parser.add_argument('--dropout', '-dp', type=float, default=0.2,
                        help='Dropout rate, methods to avoid overfitting, default: 0.3.')
    parser.add_argument('--threshold', '-t', type=float, default=0.005,
                        help='Threshold to filter edges, default: 0.0005')

    parser.add_argument('--patience', '-p', type=int, default=20, help='Patience')
    parser.add_argument('--n_repeat', type=int, default=5, help='repeat times')

    parser.add_argument('--dataset', type=str, default="BRCA", help='datastet name')
    parser.add_argument('--hidden', '-hd', type=int, default=64, help='Hidden layer dimension')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda')
    parser.add_argument('--layer', type=int, default=8, help='Number of layers.')

    parser.add_argument('--ratio', type=float, default=0.6, help='train size ratio')

    args = parser.parse_args()

    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feature_data = "result/" + args.dataset + "/latent_data.csv"
    adj_data = "result/" + args.dataset + "/PSN_matrix.csv"
    label_data = "data/" + args.dataset + "/labels.csv"
    # load input files
    adj, data, label = load_data(adj_data, feature_data, label_data, args.threshold)

    adj = torch.tensor(adj, dtype=torch.float, device=device)
    features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float, device=device)
    labels = torch.tensor(label.values, dtype=torch.long, device=device)

    nclass = len(torch.unique(labels))

    print('Begin training')
    print("ratio: {}".format(args.ratio))
    # save sample name
    all_sample = data['Sample'].tolist()

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
        DeepGCN_model = DeepGCN(nfeat=features.shape[1], nclass=nclass, dim=args.hidden,
                                alpha=args.alpha, lamda=args.lamda, nlayers=args.layer, dropout=args.dropout)
        DeepGCN_model.to(device)
        optimizer = torch.optim.Adam(DeepGCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)

        idx_train, idx_test = torch.tensor(train_idx, dtype=torch.long, device=device), torch.tensor(test_idx, dtype=torch.long, device=device)

        loss_values = []  
        acc_values = []
        # record the best epoch
        bad_counter, best_epoch = 0, 0
        best = 1000
        with tqdm(total=args.epochs, desc="Training", ncols=100) as pbar:
            for epoch in range(args.epochs):
                acc_train, loss_train = train(epoch, optimizer, features, adj, labels, idx_train)
                loss_values.append(loss_train)
                acc_values.append(acc_train)

                pbar.set_postfix(
                    {'Loss_train': '{:.3f}'.format(loss_train), 'ACC_train': '{:.2f}'.format(acc_train * 100)})
                pbar.update(1)

                if loss_values[-1] < best:
                    best = loss_values[-1]
                    best_epoch = epoch
                    bad_counter = 0
                    # weights = deepcopy(DeepGCN_model.state_dict())
                else:
                    bad_counter += 1

                if bad_counter == args.patience:
                    break

                # save model of this epoch
                torch.save(DeepGCN_model.state_dict(), './result/{}/model/{}.pkl'.format(args.dataset, epoch))

                # reserve the best model, delete other models
                files = glob.glob('./result/{}/model/*.pkl'.format(args.dataset))
                for file in files:
                    name = file.split('\\')[1]
                    epoch_nb = int(name.split('.')[0])
                    # print(file, name, epoch_nb)
                    if epoch_nb != best_epoch:
                        os.remove(file)
        # torch.save(weights, './result/{}/model/{}.pkl'.format(args.dataset, best_epoch))

        print('The best epoch is ', best_epoch)
        DeepGCN_model.load_state_dict(torch.load('./result/{}/model/{}.pkl'.format(args.dataset, best_epoch)))
        test_acc, f1, p, r = test(features, adj, labels, idx_test)

        all_ACC.append(test_acc)
        all_F1.append(f1)
        all_P.append(p)
        all_R.append(r)

    fp = open("{}_results.txt".format(args.dataset), "a+", encoding="utf-8")
    fp.write("ratio: {}\n".format(args.ratio))
    fp.write("ep: {}\n".format(args.epochs))
    fp.write("layer: {}\n".format(args.layer))
    fp.write("alpha:{}, lamda:{}\n".format(args.alpha, args.lamda))
    fp.write("wd:{}, dp:{}, lr:{}\n".format(args.weight_decay, args.dropout, args.learningrate))
    fp.write("ACC: {:.2f} ({:.2f})\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    fp.write("MaF: {:.2f} ({:.2f})\n".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
    fp.write("MaP: {:.2f} ({:.2f})\n".format(np.mean(all_P) * 100, np.std(all_P) * 100))
    fp.write("MaR: {:.2f} ({:.2f})\n\n".format(np.mean(all_R) * 100, np.std(all_R) * 100))
    fp.close()

    print("================================")
    print("Finall Test Reslut:")
    print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    print("MaF: {:.2f} ({:.2f})".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
    print("MaP: {:.2f} ({:.2f})".format(np.mean(all_P) * 100, np.std(all_P) * 100))
    print("MaR: {:.2f} ({:.2f})".format(np.mean(all_R) * 100, np.std(all_R) * 100))
    print("================================")



