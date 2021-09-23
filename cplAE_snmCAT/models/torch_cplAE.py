from numpy import matrix
import torch
import torch.nn as nn


# Convenience functions
def tensor(x, device): return torch.tensor(x).to(dtype=torch.float32).to(device)
def tensor_(x, device): return torch.as_tensor(x).to(dtype=torch.float32).to(device)
def tonumpy(x): return x.cpu().detach().numpy()


class Encoder_T(nn.Module):
    """
    Encoder for transcriptomic data

    Args:
        in_dim: input size of data
        int_dim: number of units in hidden layers
        out_dim: set to latent space dim
        dropout_p: dropout probability
    """

    def __init__(self,
                 in_dim=1000,
                 int_dim=50,
                 out_dim=3,
                 dropout_p=0.5):

        super(Encoder_T, self).__init__()
        self.drp = nn.Dropout(p=dropout_p)
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)
        self.fc4 = nn.Linear(int_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, affine=False, eps=1e-05,
                                 momentum=0.1, track_running_stats=True)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def forward(self, x):
        x = self.drp(x)
        x = self.elu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        z = self.bn(x)
        return z


class Decoder_T(nn.Module):
    """
    Decoder for transcriptomic data

    Args:
        in_dim: set to embedding dim obtained from encoder
        int_dim: number of units in hidden layers
        out_dim: number of outputs
    """

    def __init__(self,
                 in_dim=3,
                 int_dim=50,
                 out_dim=1000):

        super(Decoder_T, self).__init__()
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)
        self.Xout = nn.Linear(int_dim, out_dim)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def forward(self, x):
        x = self.fc0(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.Xout(x))
        return x


class Encoder_E(nn.Module):
    """
    Encoder for epigenetic data
    
    Args:
        in_dim: input size of data
        int_dim: number of units in hidden layers
        out_dim: set to latent space dim
        dropout_p: dropout probability
    """

    def __init__(self,
                 in_dim=1000,
                 int_dim=50,
                 out_dim=3,
                 dropout_p=0.5):

        super(Encoder_E, self).__init__()
        self.drp = nn.Dropout(p=dropout_p)
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)
        self.fc4 = nn.Linear(int_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, affine=False, eps=1e-05,
                                 momentum=0.1, track_running_stats=True)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def forward(self, x):
        x = self.drp(x)
        x = self.elu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        z = self.bn(x)
        return z


class Decoder_E(nn.Module):
    """
    Decoder for epigenetic data.

    Args:
        in_dim: set to embedding dim obtained from encoder
        int_dim: number of units in hidden layers
        out_dim: number of outputs
    """

    def __init__(self,
                 in_dim=3,
                 int_dim=50,
                 out_dim=1000):

        super(Decoder_E, self).__init__()
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)
        self.Xout = nn.Linear(int_dim, out_dim)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self, x):
        x = self.fc0(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.Xout(x))
        return x


class Model_TE(nn.Module):
    """Coupled autoencoder model

    Args:
        T_dim: Number of genes in T data
        E_dim: Number of genes in E data
        T_int_dim: hidden layer dims for T model
        E_int_dim: hidden layer dims for E model
        T_dropout: dropout for T data
        E_dropout: dropout for E data
        latent_dim: dim for representations
        alpha_T: loss weight for T reconstruction 
        alpha_E: loss weight for E reconstruction
        lambda_TE: loss weight coupling loss
        augment_decoders (bool): augment decoder with cross modal representation if True
        name: TE
    """

    def __init__(self,
                 T_dim=1000, T_int_dim=50, T_dropout=0.5,
                 E_dim=1000, E_int_dim=50, E_dropout=0.5,
                 latent_dim=3, alpha_T=1.0, alpha_E=1.0, lambda_TE=1.0,
                 augment_decoders=True,
                 T_genes=None, E_genes=None):

        super(Model_TE, self).__init__()

        self.T_dim = T_dim
        self.T_int_dim = T_int_dim
        self.T_dropout = T_dropout
        self.E_dim = E_dim
        self.E_int_dim = E_int_dim
        self.E_dropout = E_dropout
        self.latent_dim = latent_dim
        self.alpha_T = alpha_T
        self.alpha_E = alpha_E
        self.lambda_TE = lambda_TE
        self.augment_decoders = augment_decoders
        self.T_genes = T_genes
        self.E_genes = E_genes

        self.eT = Encoder_T(dropout_p=T_dropout, in_dim=T_dim, out_dim=latent_dim, int_dim=T_int_dim)
        self.eE = Encoder_E(dropout_p=E_dropout, in_dim=E_dim, out_dim=latent_dim, int_dim=E_int_dim)
        self.dT = Decoder_T(in_dim=latent_dim, out_dim=T_dim, int_dim=T_int_dim)
        self.dE = Decoder_E(in_dim=latent_dim, out_dim=E_dim, int_dim=E_int_dim)
        return

    def get_hparams(self):
        hparam_dict = {}
        hparam_dict['T_dim'] = self.T_dim
        hparam_dict['T_int_dim'] = self.T_int_dim
        hparam_dict['T_dropout'] = self.T_dropout
        hparam_dict['E_dim'] = self.E_dim
        hparam_dict['E_int_dim'] = self.E_int_dim
        hparam_dict['E_dropout'] = self.E_dropout
        hparam_dict['latent_dim'] = self.latent_dim
        hparam_dict['alpha_T'] = self.alpha_T
        hparam_dict['alpha_E'] = self.alpha_E
        hparam_dict['lambda_TE'] = self.lambda_TE
        hparam_dict['augment_decoders'] = self.augment_decoders
        hparam_dict['T_genes'] = self.T_genes
        hparam_dict['E_genes'] = self.E_genes
        return hparam_dict

    @staticmethod
    def min_var_loss(zi, zj):
        #SVD calculated over all entries in the batch
        batch_size = zj.shape[0]
        zj_centered = zj - torch.mean(zj, 0, True)
        min_eig = torch.min(torch.linalg.svdvals(zj_centered))
        min_var_zj = torch.square(min_eig)/(batch_size-1)

        zi_centered = zi - torch.mean(zi, 0, True)
        min_eig = torch.min(torch.linalg.svdvals(zi_centered))
        min_var_zi = torch.square(min_eig)/(batch_size-1)

        zi_zj_mse = torch.mean(torch.sum(torch.square(zi-zj), 1))
        loss_ij = zi_zj_mse/torch.squeeze(torch.minimum(min_var_zi, min_var_zj))
        return loss_ij

    @staticmethod
    def mean_sq_diff(x, y):
        return torch.mean(torch.square(x-y))

    def forward(self, inputs):
        #T arm forward pass
        XT = inputs[0]
        zT = self.eT(XT)
        XrT = self.dT(zT)

        #E arm forward pass
        XE = inputs[1]
        zE = self.eE(XE)
        XrE = self.dE(zE)

        #Loss calculations
        loss_dict = {}
        loss_dict['recon_T'] = self.alpha_T * self.mean_sq_diff(XT, XrT)
        loss_dict['recon_E'] = self.alpha_E * self.mean_sq_diff(XE, XrE)
        loss_dict['cpl_TE'] = self.lambda_TE * self.min_var_loss(zT, zE)

        if self.augment_decoders:
            XrT_aug = self.dT(zE.detach())
            XrE_aug = self.dE(zT.detach())
            loss_dict['recon_T_aug'] = self.alpha_T * self.mean_sq_diff(XT, XrT_aug)
            loss_dict['recon_E_aug'] = self.alpha_E * self.mean_sq_diff(XE, XrE_aug)

        return zT, zE, XrT, XrE, loss_dict


def collect_losses(loss_dict, tracked_loss):
    """Used only for logging purposes.  accumulate losses over steps (batches) in each epoch. 
    `tracked_losses`={} signifies the start of new epoch.

    Args:
        loss_dict (Dict): losses for current batch of data supplied to the model
        tracked_loss (Dict): accumulated losses. Empty Dict at the beginning of each epoch

    Returns:
        tracked_loss (Dict)
    """
    #Init if keys dont exist
    if not tracked_loss:
        tracked_loss = {key: 0 for key in loss_dict}
        tracked_loss['steps'] = 0.0

    #Add info
    tracked_loss['steps'] += 1
    for key in loss_dict:
        tracked_loss[key] += tonumpy(loss_dict[key])
    return tracked_loss
