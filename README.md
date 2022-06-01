# LTD

This is source code for vldb 2022 submission: Adaptive Distillation of Graph Neural Networks

# Appendix

We listed the detailed hyper-parameters for reproducibility in Appendix.pdf

## Requirements:

- dgl==0.6.0
- Keras==2.4.3
- numpy==1.19.2
- optuna==2.6.0
- pandas==1.2.4
- python==3.8.8
- scikit-learn==0.19.2
- scipy==1.6.2
- sklearn==0.0
- tabulate==0.8.9
- torch==1.8.1
- torchvision==0.9.1

## Data：
Because the A-Computers dataset is too large, we did not put it in the project file. The dataset can be downloaded from the following link and placed in the directory data/npz:

https://www.dropbox.com/s/26zd460xn4u6gmn/amazon_electronics_computers.npz?dl=0

## Training:

Run student model:

```
python spawn_worker.py --dataset XXXX --teacher XXX --student XXX
```

For example, if you want train GCN on cora dataset:

```
python spawn_worker.py --dataset cora --teacher GCN --student GCN
```

We fix hyper-parameters setting as reported in our paper, and you also can use `--automl` to search hyper-parameters with the help of Optuna. 
