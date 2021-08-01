# Coarse-Grained Block Pruning of Neural Network Model

## Introduction

We implemented the coarse-grained pruning strategy based on the blocky sparse structure on neural network. This method mainly focuses on fully connected layers. It can also be well combined with convolutional networks. In our method, the weight matrix is divided into square sub-blocks. Whether the sub-block is retained or pruned will depend on its importance in the network. Multiple sizes of sub-blocks as the pruning granularity are explored. After the block pruning, the weight matrix of the model becomes blocky sparse. This coarse-grained sparse structure is computationally friendly.<br>

There are three folders: `LeNet_MNIST_code`, `LeNet_FashionMNIST_code`, `ResNet_CIFAR10_code`. <br>

The codes in the `LeNet_MNIST_code` folder implement block pruning on the LeNet-300-100 network using the MNIST dataset. <br>

The codes in the `LeNet_FashionMNIST_code` folder implement block pruning on the LeNet-300-100 network using the Fashion-MNIST dataset. <br>

The codes in the `ResNet_CIFAR10_code` folder implement block pruning on the fully connected layers of ResNet18 using the Cifar10 dataset. <br>

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

In `LeNet_MNIST_code` folder, the `unpruned.py` implements the baseline network without pruning. Do <br>
```
python unpruned.py
```
The `blockpruning_nn.py` will prune iteratively the LENET-300-100 network with n as the pruning granularity. For example, do <br>
```
python blockpruning_22.py
```
The `LeNet_FashionMNIST_code` folder is similar to the `LeNet_MNIST_code` folder. <br>
In the `ResNet_CIFAR10_code` folder, the `resnet18_cifar10_unpruned.py` implements the baseline network ResNet-18 with a fully connected layer added. Do <br>
```
python resnet18_cifar10_unpruned.py
```

The `resnet18_cifar10_pruning22.py` will prune the fully connected layers in ResNet-18 with 2 as the pruning granularity. <br>
```
python resnet18_cifar10_pruning22.py
```


## Results
The relationships of sparsity-accuracy and pruning granularity-accuracy are discussed for our method. In Lenet-300-100 model, with the increase of sparsity and pruning granularity, the accuracy has a decreasing trend. <br>

After block pruning the fully-connected layers in ResNet-18 (97% of weights pruned), the test accuracy (90.73%) on the Cifar10 dataset is higher than that of (90.37%) the ResNet-18 network without pruning.
