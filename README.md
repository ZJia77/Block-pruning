# Coarse-Grained Block Pruning of Neural Network Model

## Introduction

We implemented the coarse-grained pruning strategy of neural network based on the blocky sparse structure. This method mainly focuses on fully connected layers. It can be well combined with convolutional networks. In our method, the weight matrix is divided into square sub-blocks. Whether the sub-block is retained or pruned will depend on its importance in the network. Multiple sizes of sub-blocks as the pruning granularity are explored. After the block pruning, the weight matrix of the model becomes blocky sparse. This coarse-grained sparse structure is computationally friendly.<br>

There are three folders: LeNet_MNIST_code, LeNet_FashionMNIST_code, ResNet_CIFAR10_code. <br>

The codes in the LeNet_MNIST_code folder implemented block pruning on the LeNet-300-100 network using the MNIST dataset. <br>

The codes in the LeNet_FashionMNIST_code folder implemented block pruning on the LeNet-300-100 network using the Fashion-MNIST dataset. <br>

The codes in the ResNet_CIFAR10_code folder implemented block pruning on the fully connected layers of ResNet18 using the Cifar10 dataset. <br>

## Data

[MNIST](http://yann.lecun.com/exdb/mnist/) <br>

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) <br>

[Cifar10](http://www.cs.toronto.edu/~kriz/cifar.html) <br>

## Environments

Python 3.5 <br>
Pytorch 1.3.1<br>
Numpy <br>
CUDA 11.0 <br>

## How to run

In LeNet_MNIST_code folder, the 

The relationships of sparsity-accuracy and pruning granularity-accuracy are also discussed for our method. With the increase of sparsity and pruning granularity, the accuracy has a decrease trend.
