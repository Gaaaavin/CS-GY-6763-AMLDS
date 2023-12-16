import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Argument parsing
parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-10 with different hyperparameters.')
parser.add_argument('--job_index', type=int, default=0, help='Job index from SLURM array job')
args = parser.parse_args()

# Hyperparameters
learning_rates = torch.logspace(-4, -1, 4).tolist()
weight_decays = torch.linspace(0, 1, 5).tolist()

# Determine hyperparameters based on job index
num_lr = len(learning_rates)
num_wd = len(weight_decays)
total_combinations = num_lr * num_wd

job_index = args.job_index - 1
if job_index >= total_combinations:
    raise ValueError("Job index exceeds the number of hyperparameter combinations")

lr_index = job_index // num_wd
wd_index = job_index % num_wd

learning_rate = learning_rates[lr_index]
weight_decay = weight_decays[wd_index]

# TensorBoard writer
writer = SummaryWriter(f'runs/resnet_experiment_lr_{learning_rate}_wd_{weight_decay}')

# Dataset preparation
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

# Model
net = torchvision.models.resnet18(weights=None, num_classes=10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
num_epochs = 500  # Adjust the number of epochs according to your needs
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Log training loss
    writer.add_scalar('Training Loss', running_loss / len(trainloader), epoch)

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
    writer.add_scalar('Testing Error', test_error, epoch)

# Close the writer
writer.close()

# Save model
# torch.save(net.state_dict(), f'model_index_{args.job_index}_lr_{learning_rate}_wd_{weight_decay}.pth')
