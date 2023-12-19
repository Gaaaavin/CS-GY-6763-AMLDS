import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

lr = 1e-3
wd = 1e-3

net = torchvision.models.resnet18(weights=None, num_classes=10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

wd_list = []
for e in range(100):
    for group in optimizer.param_groups:
        group['weight_decay'] = wd * (lr / group['lr'])
    wd_list.append(wd * (lr / group['lr']))
    scheduler.step()

plt.figure(dpi=120)
plt.xlabel('Epochs')
plt.ylabel('Effective Weight Decay')
plt.plot(range(100), wd_list)
plt.grid()
plt.savefig('explosion.pdf')