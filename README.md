### Coupled autoencoders snmCAT dataset


### Environment

1. Navigate to the `cplAE_MET` folder with the `setup.py` file.
2. Create the conda environment with the required dependencies.
```bash
conda create -n snmCAT
conda activate snmCAT
conda install python=3.7
conda install pytorch torchvision torchaudio -c pytorch 
pip install jupyterlab scikit-learn feather-format seaborn pandas rich tqdm timebudget autopep8 pyqt5
pip install tensorflow #Caution: tf models are deprecated.
```


3. Install the development version of this repository
```bash
pip install -e .
```

4. Install the `cplAE_TE` repository. Directly installing from github through pip doesn't work as expected. 
```bash
# can do this within any directory on local machine
git clone https://github.com/AllenInstitute/coupledAE-patchseq
cd coupledAE-patchseq
pip install -e .
```

5. Use the `data_dir` variable in `config.toml` file to specify data path. `data_dir` is expected to contain:
```
data_dir
    ├── CH.csv
    ├── mCH.csv
    ├── metadata.csv
    └── rna.csv
```

### Dataset

Human snmCAT-seq dataset from Luo et al. 2019, shared by Fangming Xie from Eran Mukamel's group.

DNA methylation (C), open chromatin (A), and transcriptomes (T) were measured in the same set of cells. 

- `metadata.csv`: general information of each cell, including ID, biological sample, and cell cluster labels.
- `rna.csv`: Unnormalized  `cell x gene` count matrix
- `mCH.csv`: `cell x gene` matrix, representing methylated cytosine count in the gene body of that gene
- `CH.csv`: `cell x gene` matrix with total number of cytosines (methylated + unmethylated). 

`mCH` and `CH` matrices have the same dimensions. Element-wise ratio of `mCH` to `CH` (i.e. fraction of methylated to total cytosines) are usually interpreted as DNA methylation levels.