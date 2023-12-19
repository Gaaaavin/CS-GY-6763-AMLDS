
import random
import argparse
import numpy as np
from tqdm import trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# hyper-parameters
LR = 1e-3
NUM_EPOCHS = 100
WD_LIST = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):
    # Check for all weight decay factors
    for i, wd in enumerate(WD_LIST):

        # Init exp
        print(f'\nExp: optimizer = {args.optimizer}, wd = {wd:.6f}, seed = {args.seed}')
        seed_everything(args.seed)
        exp_dict = defaultdict(list)            

        # Dataset preparation
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=f'./data_{i}', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(root=f'./data_{i}', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

        # Model
        net = torchvision.models.resnet18(weights=None, num_classes=10)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = net.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        if args.optimizer == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=wd)
        elif args.optimizer in ('adamw', 'adamw_fwd'):
            optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=wd)
        else:
            raise ValueError(f'Invalid optimizer: {args.optimizer}')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        # Training loop
        for _ in trange(NUM_EPOCHS):

            # Adjust weight decay if needed
            if args.optimizer == 'adamw_fwd':
                for group in optimizer.param_groups:
                    group['weight_decay'] = wd * (LR / group['lr'])

            # Perform Training
            net.train()
            running_loss = 0.0
            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Log training loss
            exp_dict['train_loss'].append(running_loss / len(trainloader))

            # Evaluate on testing data
            net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_error = 1 - correct / total
            exp_dict['test_error'].append(test_error)

            # Update scheduler
            scheduler.step()

        # Save exp results
        output_path = f'exp/{args.optimizer}_wd{i}_{args.seed}.pt'
        torch.save(exp_dict, output_path)
        print(f'Results saved to: {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'adamw_fwd'], required=True)
    parser.add_argument('--seed', type=int, default=0, required=True)
    args = parser.parse_args()
    main(args)
