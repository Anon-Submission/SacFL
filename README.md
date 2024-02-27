# Code for SacFL
Code for the paper "SacFL: Self-Adaptive Federated Continual Learning towards Resource-Constrained End Devices"


## Requirements
- python 3.7+
- cuda 9.0+
- pytorch 1.7+


## Datasets
- FashionMNIST
- CIFAR-10
- CIFAR-100
- THUCNews

To construct the required data, we provide some tools in `` dataset/data_process.py ``.

# Benchmarks
- CFeD: Continual federated learning based on knowledge distillation
- LwF-Fed: Continual federated learning based on knowledge distillation
- EWC-Fed: Overcoming catastrophic forgetting in neural networks
- MultiHead-Fed: Continual federated learning based on knowledge distillation
- FedAvg: Continual federated learning based on knowledge distillation
- FedProx: Federated optimization in heterogeneous networks
