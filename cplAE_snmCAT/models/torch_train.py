import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from cplAE_snmCAT.models.torch_cplAE import Model_TE, collect_losses, tonumpy, tensor_
from cplAE_snmCAT.utils.load_config import load_config
from cplAE_snmCAT.utils.proc import get_splits, select_dataset_v1
from torch.utils.data import DataLoader
from functools import partial


parser = argparse.ArgumentParser()
parser.add_argument("--batchsize",         default=500,             type=int,    help="Batch size")
parser.add_argument("--alpha_T",           default=1.0,             type=float,  help="Transcriptomic reconstruction loss weight")
parser.add_argument("--alpha_E",           default=1.0,             type=float,  help="Epigenetic reconstruction loss weight")
parser.add_argument("--lambda_TE" ,        default=1.0,             type=float,  help="Coupling loss weight")
parser.add_argument("--augment_decoders",  default=1,               type=int,    help="0 or 1 : Train with cross modal reconstruction")
parser.add_argument("--latent_dim",        default=3,               type=int,    help="Number of latent dims")
parser.add_argument("--n_epochs",          default=5000,            type=int,    help="Number of epochs to train")
parser.add_argument("--remove_outliers",   default=1,               type=int,    help="Treated as boolean")
parser.add_argument("--run_iter",          default=0,               type=int,    help="Run-specific id")
parser.add_argument("--model_id",          default='TE',            type=str,    help="Model-specific id")
parser.add_argument("--exp_name",          default='snmCAT_torch',  type=str,    help="Experiment set")


def set_paths(exp_name='TEMP'):
    paths = load_config(verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['logs'] = f'{paths["result"]}logs/'
    Path(paths['logs']).mkdir(parents=True, exist_ok=True)
    return paths


class TE_Dataset(torch.utils.data.Dataset):
    def __init__(self, T_dat, E_dat):
        super(TE_Dataset).__init__()
        self.T_dat = T_dat
        self.E_dat = E_dat

    def __getitem__(self, idx):
        return {'XT': self.T_dat[idx, :], 'XE': self.E_dat[idx, :]}

    def __len__(self):
        return np.shape(self.T_dat)[0]


def main(alpha_T=1.0, alpha_E=1.0, lambda_TE=1.0, augment_decoders=1.0,
         batchsize=500, latent_dim=3, n_epochs=5000, remove_outliers=1,
         run_iter=0, model_id='TE', exp_name='snmCAT_torch'):

    dir_pth = set_paths(exp_name=exp_name)
    fileid = (model_id +
              f'_aT_{str(alpha_T)}_aE_{str(alpha_E)}_' +
              f'csTE_{str(lambda_TE)}_ad_{str(augment_decoders)}_' +
              f'ld_{latent_dim:d}_bs_{batchsize:d}_ne_{n_epochs:d}_' +
              f'out_{remove_outliers:d}_' +
              f'ri_{run_iter:d}').replace('.', '-')

    #Convert int to boolean
    augment_decoders = augment_decoders > 0

    #Data selection===================
    n_genes = 1000
    D = select_dataset_v1(n_genes,
                          select_T='sorted_highvar_T_genes',
                          select_E='sorted_highvar_E_genes')
    train_ind, val_ind = get_splits(data=D, fold=0, n_folds=10)
    splits = {'train_ind': train_ind, 'val_ind': val_ind}

    #Helpers ==========================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    astensor = partial(tensor_,device=device)

    def report_losses(loss_dict, partition):
        for key in loss_dict.keys():
            if key!='steps':
                loss_dict[key] = loss_dict[key]/loss_dict['steps']
        return '   '.join([f'{partition}_{key} : {loss_dict[key]:0.5f}' for key in loss_dict])

    def save_results(model, data, fname, splits=splits):
        model.eval()
        zT, zE, XrT, XrE, _ = model((astensor(data['XT']), astensor(data['XE'])))
        savemat = {'zT': tonumpy(zT), 'XrT': tonumpy(XrT),
                   'zE': tonumpy(zE), 'XrE': tonumpy(XrE)}
        savemat.update(splits)
        sio.savemat(fname, savemat, do_compression=True)
        return

    #Training data
    if remove_outliers != 0:
        outliers_df = pd.read_csv(dir_pth['data_dir'] / 'outliers.csv')
        outliers_ctype_df = pd.read_csv(dir_pth['data_dir'] / 'outliers_ctype.csv')
        train_ind = np.setdiff1d(train_ind, outliers_df['ind'].values)
        train_ind = np.setdiff1d(train_ind, outliers_ctype_df['ind'].values)
    train_dataset = TE_Dataset(T_dat=D['XT'][train_ind, :], E_dat=D['XE'][train_ind, :])
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False,
                                  sampler=train_sampler, drop_last=True, pin_memory=True)


    #Model ============================
    model = Model_TE(T_dim=n_genes, T_int_dim=50, T_dropout=0.2,
                     E_dim=n_genes, E_int_dim=50, E_dropout=0.2,
                     latent_dim=latent_dim, alpha_T=alpha_T, alpha_E=alpha_E, lambda_TE=lambda_TE,
                     augment_decoders=augment_decoders,
                     T_genes=D['genesT'], E_genes=D['genesE'])

    optimizer = torch.optim.Adam(model.parameters())

    model.to(device)
    #Training loop ====================
    best_loss = np.inf
    best_epoch = 0
    monitor_loss = []
    
    for epoch in range(n_epochs):
        train_loss_dict = {}
        train_datagen = iter(train_dataloader)
        for _ in range(len(train_dataloader)):
            batch = next(train_datagen)
            
            #zero + forward + backward + udpate
            optimizer.zero_grad()
            zT, zE, XrT, XrE, loss_dict = model((astensor(batch['XT']),astensor(batch['XE'])))
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()
            
            #track loss over batches:
            train_loss_dict = collect_losses(loss_dict=loss_dict, tracked_loss=train_loss_dict)

        #Validation: train mode -> eval mode + no_grad + eval mode -> train mode
        model.eval()
        with torch.no_grad():
            zT, zE, XrT, XrE, loss_dict = model((astensor(D['XT'][val_ind, :]),astensor(D['XE'][val_ind, :])))
        val_loss_dict = collect_losses(loss_dict=loss_dict, tracked_loss={})
        model.train()
        
        train_loss = report_losses(loss_dict=train_loss_dict, partition="train")
        val_loss = report_losses(loss_dict=val_loss_dict, partition="val")
        print(f'epoch {epoch:04d} Train {train_loss}')
        print(f'epoch {epoch:04d} ----- Val {val_loss}')

        #Logging ==============
        with open(dir_pth['logs'] + f'{fileid}.csv', "a") as f:
            writer = csv.writer(f, delimiter=',')
            if epoch == 0:
                writer.writerow(['epoch', *['train_' + s for s in train_loss_dict.keys()], *['val_' + s for s in val_loss_dict.keys()]])
            writer.writerow([epoch+1, *train_loss_dict.values(), *val_loss_dict.values()])
        monitor_loss.append(val_loss_dict['recon_T']+val_loss_dict['recon_E']+val_loss_dict['cpl_TE'])

        #Checkpoint ===========
        if (monitor_loss[-1] < best_loss) and (epoch - best_epoch > 500):
            best_loss = monitor_loss[-1]
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'hparams': model.get_hparams(),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss, }, dir_pth['result'] + f'{fileid}-best_weights.pt')
    
    print('\nTraining completed.')
    #Save model weights on exit
    torch.save({'epoch': epoch,
                'hparams': model.get_hparams(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss}, dir_pth['result'] + f'{fileid}_exit_weights.pt')

    save_fname = dir_pth['result'] + f'{fileid}_exit'
    save_results(model=model, data=D.copy(), fname=f'{save_fname}-summary.mat')
    print('\nSaved weights and summary.')
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
