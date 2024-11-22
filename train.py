import torch
import torch.nn as nn
import torch.optim as optim

from utils import count_parameters, get_model, get_num_classes, get_dataloader

import argparse
import logging
import datetime
import json


def train(model, dataloader, num_epochs, device, logger, filename):

    logger.info(f"Start Training for {num_epochs} epochs.")
    logger.info(f"Training with device: {device}.")

    model.to(device)
    train_loader, test_loader = dataloader

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {
        'train_loss'        : [],
        'train_accuracy'    : [],
        'test_loss'         : [],
        'test_accuracy'     : []
    }

    # Training Loop
    for epoch in range(num_epochs):
        train_start_time = datetime.datetime.now()
        model.train()
        running_loss = 0.0
        correct=0
        total=0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss/len(train_loader)
        train_accuracy = 100 * correct / total
        train_time = datetime.datetime.now() - train_start_time
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Training Accuracy : {train_accuracy}, Train Time {train_time}")

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        test_start_time = datetime.datetime.now()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        test_loss = running_loss/len(train_loader)
        test_accuracy = 100 * correct / total
        test_time = datetime.datetime.now()-test_start_time

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}, Test Time {test_time}")

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)

    torch.save(model.state_dict(), f'{filename}.pth')
    with open(f"{filename}.json", "w") as outfile:
        json.dump(history, outfile, indent=4)

    return model, history

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--adaact', default=False, type=bool)
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--epochs', default=100, type = int)
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--seed', default=69, type=int, help='Random seed.')

    args = parser.parse_args()


    # Create and configure logger
    filename=f"./checkpoints/{args.model}_adaact{args.adaact}_{args.dataset}_epochs{args.epochs}"
    logging.basicConfig(filename=f"{filename}.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(f"Args: {args}")

    torch.manual_seed(args.seed)


    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        args.gpu_index = -1

    with torch.cuda.device(args.gpu_index):
        
        model = get_model(args.model, get_num_classes(args.dataset), args.adaact).to(device)
        logger.info(f'Created {args.model} model with parameters {count_parameters(model)}')
        logger.info(model)
        dataloader = get_dataloader(args.dataset)
        model, history = train(model, dataloader, args.epochs, device, logger, filename)