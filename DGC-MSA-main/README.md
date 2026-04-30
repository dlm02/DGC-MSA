# DGC-MSA

**Dual-View Graph Contrastive Clustering of scRNA-seq Data via Multi-Scale Attention**

DGC-MSA is a robust deep learning framework specifically designed for clustering single-cell RNA sequencing (scRNA-seq) data. To effectively capture complex cellular heterogeneity and mitigate structural noise, the model integrates multi-scale graph neural networks, gated cross-attention mechanisms, and dual-view contrastive learning.

## Requirements

Recommended environment:

```text
Python 3.8
PyTorch 1.11.0
Scanpy 1.8.2
NumPy 1.19.5
SciPy 1.6.2
scikit-learn
pandas
anndata
h5py
leidenalg
umap-learn
matplotlib
```

Install dependencies:

```bash
conda create -n dgcmsa python=3.8
conda activate dgcmsa
pip install torch torchvision torchaudio
pip install scanpy anndata pandas numpy scipy scikit-learn h5py leidenalg umap-learn matplotlib
```

## File Description

| File | Description |
| --- | --- |
| `model.py` | Defines the main DGC-MSA model, including the autoencoder, graph neural network, attention fusion module, and contrastive projection head. |
| `loss.py` | Defines ZINB/NB reconstruction losses and contrastive loss. |
| `train.py` | Contains model training and clustering optimization. |
| `utils.py` | Contains data loading, preprocessing, graph construction, clustering initialization, and visualization functions. |
| `run_DGC-MSA.py` | Main script for running the model. |
| `preprocessing_h5.py` | Converts `.h5` datasets into `.h5ad` files. |
| `preprocessing_baron.py` | Converts Baron pancreas datasets into `.h5ad` files. |

## Data Preparation

Put the scRNA-seq dataset in the following directory:

```text
./Data/AnnData/
```

The input file should be in `.h5ad` format:

```text
./Data/AnnData/{dataset_name}.h5ad
```

The expression matrix should be stored in:

```python
adata.X
```

If true cell labels are available, store them in:

```python
adata.obs["celltype"]
```

## Usage

Run the model with:

```bash
python run_DGC-MSA.py --name Muraro --celltype known
```

Run on GPU:

```bash
python run_DGC-MSA.py --name Muraro --celltype known --cuda True
```

If true cell labels are unavailable:

```bash
python run_DGC-MSA.py --name Muraro --celltype unknown
```

## Main Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--name` | `Romanov` | Dataset name. |
| `--lr` | `0.001` | Learning rate. |
| `--n_z` | `16` | Latent embedding dimension. |
| `--n_heads` | `8` | Number of attention heads. |
| `--n_hvg` | `2500` | Number of highly variable genes. |
| `--training_epoch` | `200` | Number of pretraining epochs. |
| `--clustering_epoch` | `100` | Number of clustering epochs. |
| `--resolution` | `1.0` | Leiden clustering resolution. |
| `--connectivity_methods` | `gauss` | Graph construction method. |
| `--n_neighbors` | `15` | Number of neighbors for graph construction. |
| `--cuda` | `False` | Whether to use GPU. |

## Output

If true labels are provided, the program reports clustering metrics including:

```text
ARI, NMI, ASW, DB
```

The program can also save prediction labels, embeddings, UMAP figures, and model parameters in the following folders:

```text
./pred_label/
./embedding/
./umap_figure/
./model_save/
```
