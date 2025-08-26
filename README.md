# GTA_Assignments

This repository contains two simple Python implementations for:
- PageRank algorithm
- Node2Vec algorithm

## PageRank Implementation in Python

It demonstrates how PageRank can be computed iteratively using a transition matrix 
and the power iteration method.

### 📌 Features
- Pure Python + NumPy implementation of PageRank
- Adjustable damping factor
- Iterative computation with convergence criterion
- Example with adjacency matrix input
- Prints the PageRank values and number of iterations to convergence

## Node2Vec from Scratch with Skip-gram


Minimal Python implementation of **node2vec** embeddings
for small graphs, using **random walks** and a **Skip-gram model with Negative Sampling (SGNS)**.  

---

### 📌 Features

- Build random walks on arbitrary graphs with biased sampling (p, q parameters).  
- Extract center-context pairs from walks.  
- Train Skip-gram with Negative Sampling using PyTorch.  
- Obtain low-dimensional embeddings for nodes.
- Visualize nodes embeddings

---

## 🛠 Requirements
The implementations relies on the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `torch`
- `networkx`
- `random`

You can install them with:

```bash
pip install -r requirements.txt
