### Coupled autoencoders snmCAT dataset


### Environment

1. Create the conda environment with the required dependencies.
```bash
conda create -n snmCAT
conda activate snmCAT
conda install python=3.7
conda install pytorch torchvision torchaudio -c pytorch
pip install jupyterlab scikit-learn feather-format seaborn pandas rich tqdm timebudget autopep8 pyqt5
```
2. Clone and install this repo:
```
git clone https://github.com/rhngla/cplAE_snmCAT
cd cplAE_snmCAT
pip install -e .
```

3. Get the data required for this repository
```
data_dir
    ├── CH.csv
    ├── mCH.csv
    ├── metadata.csv
    └── rna.csv
```

4. Specify data paths in `config.toml` (at repository root level):
```toml
data_dir = "/home/Local/dat/raw/snmCAT-seq/"
metadata_file = "/home/Local/dat/raw/snmCAT-seq/metadata.csv"
rna_file = "/home/Local/dat/raw/snmCAT-seq/rna.csv"
mCH_file = "/home/Local/dat/raw/snmCAT-seq/mCH.csv"
CH_file = "/home/Local/dat/raw/snmCAT-seq/CH.csv"
outlier_file = "/home/Local/dat/raw/snmCAT-seq/outliers.csv" # .csv is generated within notebook 02_ouliers_E.ipynb
```

### Dataset description

Human snmCAT-seq dataset from Luo et al. 2019, shared by Fangming Xie from Eran Mukamel's group.

DNA methylation (mC), open chromatin (A), and transcriptomes (T) were measured in the same set of single nuclei .

- `metadata.csv`: general information of each cell, including ID, biological sample, and cell cluster labels.
- `rna.csv`: Unnormalized  `cell x gene` count matrix
- `mCH.csv`: `cell x gene` matrix, representing methylated cytosine count in the gene body of that gene
- `CH.csv`: `cell x gene` matrix with total number of cytosines (methylated + unmethylated).

`mCH` and `CH` matrices have the same dimensions. Element-wise ratio of `mCH` to `CH` (i.e. fraction of methylated to total cytosines) are usually interpreted as DNA methylation levels.