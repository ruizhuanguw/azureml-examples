# Comparing AutoMLs

description: comparing various AutoML tools

non-exhaustive list to test:

- Azure AutoML
- autosklearn
- PyCaret
- TPOT
- Hypernets
- ...

## Getting started

PyCaret and Azure AutoML are incompatible. Create different conda envs:

```console
conda create -n pycaret python=3.8
conda activate pycaret
pip install --upgrade ipykernel ipython jupyter
```

```console
conda create -n aautoml python=3.6
conda activate aautoml
pip install --upgrade ipykernel ipython jupyter
```
