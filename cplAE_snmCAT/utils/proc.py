import numpy as np
import pandas as pd
import feather
import scipy.io as sio
from cplAE_snmCAT.utils.load_config import load_config

def proc_dataset_v1(write=False):
    """pre-process and write for the paired transcriptomics and epigenetic data
    and metadata files.

    Returns:
        T: dataframe with log1p(CPM) normalized expression
        E: dataframe with Epigenetic data
        M: dataframe containing Metadata
        data: arrays of genes sorted by variance of the processed matrices
    """
    
    path = load_config()
    M = pd.read_csv(path['metadata_file'])
    T = pd.read_csv(path['rna_file'])
    mCH = pd.read_csv(path['mCH_file'])
    CH = pd.read_csv(path['CH_file'])

    def format_df(df):
        """The inputs are genes x cells. Transpose data and rename columns"""
        df = df.transpose()
        df.rename(columns=df.iloc[0], inplace=True)
        df.drop('gene', inplace=True)
        df.index.rename('sample_id', inplace=True)
        return df

    T = format_df(T)
    mCH = format_df(mCH)
    CH = format_df(CH)

    #Update metadata
    M = pd.read_csv(path['metadata_file'])
    M.rename(columns={'Unnamed: 0': 'sample_id'}, inplace=True)
    M.set_index(keys='sample_id', drop=True, inplace=True)

    #Sort cells by metadata
    sorted_index = M.sort_values(by='SubClusterAnno').index
    M = M.loc[sorted_index]
    T = T.loc[sorted_index]
    mCH = mCH.loc[sorted_index]
    CH = CH.loc[sorted_index]

    assert np.array_equal(CH.columns, mCH.columns), "Genes are not in the same order"
    assert np.array_equal(T.columns, mCH.columns), "Genes are not in the same order"
    assert M.index.equals(T.index), "Cells are not in the same order"
    assert M.index.equals(CH.index), "Cells are not in the same order"
    assert M.index.equals(mCH.index), "Cells are not in the same order"

    # CPM-normalize counts
    X = T.values.astype(np.float32)
    X = (X/np.sum(X, axis=1, keepdims=True))*1e6
    X = np.log1p(X)
    T[T.columns] = X

    # For methylation data
    X = CH.values.astype(np.float)
    Y = mCH.values.astype(np.float)
    X = np.log1p(Y) - np.log1p(X)
    E = mCH.copy(deep=True)
    E[E.columns] = X

    # select genes based on variance in log normalized CPM values in T
    def calc_highvar_genes(df):
        vars = np.var(df.values, axis=0)
        order_vars = np.argsort(-vars)  # descending order
        sorted_highvar_genes = df.columns.values[order_vars]
        return sorted_highvar_genes

    data = {'sorted_highvar_T_genes': calc_highvar_genes(T),
            'sorted_highvar_E_genes': calc_highvar_genes(E)}

    if write:
        feather.write_dataframe(T, path['data_dir'] / 'T_dat.feather')
        feather.write_dataframe(E, path['data_dir'] / 'E_dat.feather')
        feather.write_dataframe(M, path['data_dir'] / 'Meta.feather')
        sio.savemat(path['data_dir'] / 'highvar_genes.mat', data)
    return T, E, M, data


def read_dataset_v1():
    """loads v1 of the dataset

    Returns:
        T: dataframe with log1p(CPM) normalized expression
        E: dataframe with Epigenetic data
        M: dataframe containing Metadata
        data: arrays of genes sorted by variance of the processed matrices
    """
    path = load_config()
    T = feather.read_dataframe(path['data_dir'] / 'T_dat.feather')
    E = feather.read_dataframe(path['data_dir'] / 'E_dat.feather')
    M = feather.read_dataframe(path['data_dir'] / 'Meta.feather')
    data = sio.loadmat(path['data_dir'] / 'highvar_genes.mat', squeeze_me=True)
    return T, E, M, data


def select_dataset_v1(n_genes, select_T='sorted_highvar_T_genes', select_E='sorted_highvar_E_genes'):
    T_df, E_df, M_df, genes = read_dataset_v1()
    D = {}
    D['genesT'] = genes[select_T][0:n_genes]
    D['genesE'] = genes[select_E][0:n_genes]
    D['XT'] = T_df[D['genesT']].values
    D['XE'] = E_df[D['genesE']].values
    D['cluster'] = M_df['SubClusterAnno'].values
    return D


def get_splits(data, fold, n_folds=10):
    """Creates `n_fold` cross validation sets (fixed random seed), 
    and returns the train/validation indices for the selected fold
    
    Returns:
        (train_ind,val_ind)
    Arguments:
        data: Dictionary with key 'cluster' used to stratify data
        fold: fold id
        n_folds: number of splits
    """
    assert fold < n_folds, f'fold value must be less than n_folds value ({n_folds:d})'
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, random_state=0, shuffle=True)
    all_folds = [{'train_ind': train_ind, 'val_ind': val_ind} for train_ind, val_ind in skf.split(
        X=np.zeros(shape=data['cluster'].shape), y=data['cluster'])]
    return all_folds[fold]['train_ind'], all_folds[fold]['val_ind']
