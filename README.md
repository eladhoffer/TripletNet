Deep Metric Learning Using Triplet Network
==========================================
This code replicates the results from the paper “Deep metric learning using Triplet network” (http://arxiv.org/abs/1412.6622).

It can train a TripletNet on any of the {Cifar10/100, STL10, SVHN, MNIST} datasets.

## Data
You can get the needed data using the following repos:
* CIFAR10/100: https://github.com/soumith/cifar.torch.git
* STL10: https://github.com/eladhoffer/stl10.torch
* SVHN: https://github.com/torch/tutorials/blob/master/A_datasets/svhn.lua
* MNIST: https://github.com/andresy/mnist

## Dependencies
* Torch (http://torch.ch)
* "eladtools" (https://github.com/eladhoffer/eladtools) for optimizer.
* "nngraph" (https://github.com/torch/nngraph) for TripletNet configuration.
* "cudnn.torch" (https://github.com/soumith/cudnn.torch) for faster training. Can be avoided by changing "cudnn" to "nn" in models.

## Models
Available models are at the “Models” directory. The basic Model.lua was used in the paper, while NiN based models achieve slightly better
results.

## Training
You can start training using:
```lua
th Main.lua -dataset Cifar10 -LR 0.1 -save new_exp_dir
```

## Additional flags
|Flag             | Default Value        |Description
|:----------------|:--------------------:|:----------------------------------------------
|modelsFolder     |  ./Models/           | Models Folder
|network          |  Model.lua           | Model file - must return valid network.
|LR               |  0.1                 | learning rate
|LRDecay          |  0                   | learning rate decay (in # samples
|weightDecay      |  1e-4                | L2 penalty on the weights
|momentum         |  0.9                 | momentum
|batchSize        |  128                 | batch size
|optimization     |  sgd                 | optimization method
|epoch            |  -1                  | number of epochs to train (-1 for unbounded)
|threads          |  8                   | number of threads
|type             |  cuda                | float or cuda
|devid            |  1                   | device ID (if using CUDA)
|load             |  none                | load existing net weights
|save             |  time-identifier     | save directory
|dataset          |  Cifar10             | Dataset - Cifar10, Cifar100, STL10, SVHN, MNIST
|normalize        |  1                   | 1 - normalize using only 1 mean and std values
|whiten           |  false               | whiten data
|augment          |  false               | Augment training data
|preProcDir       |  ./PreProcData/      | Data for pre-processing (means,Pinv,P)
