import torch.nn as nn
from torch.nn.functional import normalize
import torch
import math


class DPN(nn.Module):
    def __init__(self, band):
        super(DPN, self).__init__()

        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Spectral path
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3), kernel_size=(1, 1, 7),
                                stride=(1, 1, 1))
        self.conv13 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3), kernel_size=(1, 1, 7),
                                stride=(1, 1, 1))
        self.conv14 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3), kernel_size=(1, 1, 7),
                                stride=(1, 1, 1))
        self.conv15 = nn.Conv3d(in_channels=60, out_channels=12, padding=(0, 0, 3), kernel_size=(1, 1, 7),
                                stride=(1, 1, 1))
        kernel_3d = math.ceil((band - 6) / 2)
        self.conv16 = nn.Conv3d(in_channels=72, out_channels=72, kernel_size=(1, 1, kernel_3d),
                                stride=(1, 1, 1))
        self.batch_norm11 = nn.ReLU()  # Don't mind this name
        self.batch_norm12 = nn.ReLU()
        self.batch_norm13 = nn.ReLU()
        self.batch_norm14 = nn.ReLU()
        self.batch_norm15 = nn.ReLU()

        # Spatial path
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, band), stride=(1, 1, 1))
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1),
                                stride=(1, 1, 1))
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1),
                                stride=(1, 1, 1))
        self.conv24 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1),
                                stride=(1, 1, 1))
        self.conv25 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1),
                                stride=(1, 1, 1))
        self.global_pooling = nn.AdaptiveAvgPool3d(1)

        self.batch_norm21 = nn.ReLU()
        self.batch_norm22 = nn.ReLU()
        self.batch_norm23 = nn.ReLU()
        self.batch_norm24 = nn.ReLU()

        self.batch_norm_all = nn.Sequential(nn.BatchNorm3d(84, eps=0.001, momentum=0.1, affine=True),
                                            nn.ReLU())
        self.inp = 84

    def forward(self, X, Y):
        # spectral
        x11 = self.conv11(Y)

        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)

        x13 = torch.cat((x11, x12), dim=1)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        x15 = self.batch_norm14(x15)
        x15 = self.conv15(x15)

        x16 = torch.cat((x11, x12, x13, x14, x15), dim=1)

        x17 = self.batch_norm15(x16)
        x17 = self.conv16(x17)

        # Spatial
        x21 = self.conv21(X)

        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x23, x24), dim=1)
        x25 = self.batch_norm24(x25)
        x25 = self.conv25(x25)

        # Spatial & spectral
        Spatial = self.global_pooling(x25)
        spectral = x17
        x3 = torch.cat((spectral, Spatial), dim=1)

        output = self.batch_norm_all(x3)
        output = self.global_pooling(output)
        output = output.squeeze(-1).squeeze(-1).squeeze(-1)

        return output


class SimCLRStage1(nn.Module):

    def __init__(self, model):
        super(SimCLRStage1, self).__init__()
        self.model = model

        self.mlp = nn.Sequential(
            nn.Linear(model.inp, 64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x_i, x_j, y_i, y_j):
        h_i = self.model(x_i, y_i)
        h_j = self.model(x_j, y_j)

        z_i = normalize(self.mlp(h_i), dim=1)
        z_j = normalize(self.mlp(h_j), dim=1)

        return z_i, z_j


class SimCLRStage2(nn.Module):

    def __init__(self, model):
        super(SimCLRStage2, self).__init__()
        # encoder
        self.model = model
        self.detection = nn.Sequential(
            nn.Linear(model.inp, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        x = self.model(x, y)
        x = self.detection(x)

        return x



class NT_Xent_loss(nn.Module):
    def __init__(self, temperature):
        super(NT_Xent_loss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]

        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        positives = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temperature)
        molecule = torch.cat([positives, positives], dim=0)
        denominator = sim_matrix.sum(dim=-1)

        loss = (- torch.log(molecule / denominator)).mean()

        return loss
