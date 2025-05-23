# AdaAct
An adaptive activation function that combines multiple activation functions with data-driven weights, enhancing flexibility and generalization across tasks. Includes regularization for sparsity, choosing only the most relevant activations. 

## Training the models

Create a directory named ```checkpoints``` at the project root. Model weights, logs and metrics will be saved in this directory.

for using AdaAct, use the flag ```--adaact True``` in the training command.

### Training Commands

ResNet18 with CIFAR10: ```python train.py --model resnet18 --dataset cifar10```

ResNet18 with CIFAR100: ```python train.py --model resnet18 --dataset cifar100```

Vision Transformer with CIFAR10: ```python train.py --model vision_transformer --dataset cifar10```

Vision Transformer with CIFAR100: ```python train.py --model vision_transformer --dataset cifar100```


## Results on ResNet18

### CIFAR10
![alt text](media/resnet18_cifar10_epochs100.png)

### CIFAR100
![alt text](media/resnet18_cifar100_epochs100.png)


## Results on Vision Transformer

### CIFAR10
![alt text](media/vision_transformer_cifar10_epochs100.png)

### CIFAR100
![alt text](media/vision_transformer_cifar100_epochs100.png)
