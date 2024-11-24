import torch
import torch.nn as nn
from torchvision import datasets, transforms

from resnet import ResNet18
from visiontransformer import VisionTransformer

all_acts = [
    nn.ReLU(),
    nn.LeakyReLU(),
    nn.PReLU(),
    nn.ELU(),
    nn.SELU(),
    nn.GELU(),
    nn.Tanh(),
    nn.Sigmoid(),
    nn.Softplus(),
    nn.Softsign(),
    nn.SiLU(),
    nn.Hardswish(),
    nn.Hardtanh(),
    nn.Hardsigmoid(),
    nn.LogSigmoid(),
    nn.Threshold(0.5, 0.0),
    nn.Softmax(dim=-1),
    nn.LogSoftmax(dim=-1)
]


def get_num_classes(dataset):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100


def get_model(model_name, num_classes, adaact):
    if model_name == 'resnet18':
        if adaact:
            model = ResNet18(num_classes, all_acts)
        else:
            model = ResNet18(num_classes)
    if model_name == 'visionTransformer':
        if adaact:
            model = VisionTransformer(num_classes, all_acts)
        else:
            model = VisionTransformer(num_classes)

    return model


def get_dataloader(dataset):
    # CIFAR-100 Dataset and DataLoader
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    return train_loader, test_loader


def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params