import torch
from torch import nn
from matplotlib import pyplot as plt

class AE(nn.Module):
    def __init__(self, in_feas_dim, latent_dim, a=0.4, b=0.5, c=0.1):
        super(AE, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.in_feas = in_feas_dim
        self.latent = latent_dim

        # encoders
        self.encoder1 = nn.Sequential(
            nn.Linear(self.in_feas[0], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid()
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(self.in_feas[1], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid()
        )

        self.encoder3 = nn.Sequential(
            nn.Linear(self.in_feas[2], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid()
        )

        # decoders
        self.decoder1 = nn.Sequential(nn.Linear(self.latent, self.in_feas[0]))
        self.decoder2 = nn.Sequential(nn.Linear(self.latent, self.in_feas[1]))
        self.decoder3 = nn.Sequential(nn.Linear(self.latent, self.in_feas[2]))

        for name, param in AE.named_parameters(self):
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)

    def forward(self, omics_1, omics_2, omics_3):
        encoded_omics_1 = self.encoder1(omics_1)
        encoded_omics_2 = self.encoder2(omics_2)
        encoded_omics_3 = self.encoder3(omics_3)
        latent_data = torch.mul(encoded_omics_1, self.a) + torch.mul(encoded_omics_2, self.b) + torch.mul(encoded_omics_3, self.c)
        decoded_omics_1 = self.decoder1(latent_data)
        decoded_omics_2 = self.decoder2(latent_data)
        decoded_omics_3 = self.decoder3(latent_data)

        return latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3

    def train_AE(self, train_loader, learning_rate=0.001, device=torch.device('cpu'), epochs=100, dataset="TCGA"):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()
        loss_ls = []
        for epoch in range(epochs):
            train_loss_sum = 0.0  # Record the loss of each epoch
            for (x, y) in train_loader:
                omics_1 = x[:, :self.in_feas[0]]
                omics_2 = x[:, self.in_feas[0]: self.in_feas[0] + self.in_feas[1]]
                omics_3 = x[:, self.in_feas[0] + self.in_feas[1] : self.in_feas[0] + self.in_feas[1] + self.in_feas[2]]

                omics_1 = omics_1.to(device)
                omics_2 = omics_2.to(device)
                omics_3 = omics_3.to(device)

                latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3 = self.forward(omics_1, omics_2, omics_3)
                loss = self.a * loss_function(decoded_omics_1, omics_1) + self.b * loss_function(decoded_omics_2, omics_2) + self.c * loss_function(
                    decoded_omics_3, omics_3)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.sum().item()

            loss_ls.append(train_loss_sum)
            print('epoch: %d, loss: %.4f' % (epoch + 1, train_loss_sum))

        # draw the training loss curve
        plt.plot([i + 1 for i in range(epochs)], loss_ls)
        plt.xlabel('Epochs', fontsize=12, fontname='Times New Roman')
        plt.ylabel('Loss', fontsize=12, fontname='Times New Roman')
        plt.savefig('result/{}/AE_train_loss.png'.format(dataset))
