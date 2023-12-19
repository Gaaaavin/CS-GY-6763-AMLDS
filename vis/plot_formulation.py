
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

num_exp = 5
num_epoch = 100
optimizer_list = ['adam', 'adamw', 'adamw_fwd']

fig, ax = plt.subplots(2, 3, figsize=(12, 8), dpi=120)
plt.subplots_adjust(hspace=0.35)
plt.subplots_adjust(wspace=0.25)

for wd in range(6):

    wd_group = f'wd{wd}'

    # Load training logs
    logs_dict = {}
    for opt in optimizer_list:
        for i in range(num_exp):
            exp_id = f'{opt}_{wd_group}_{i}'
            logs_dict[exp_id] = torch.load(f'../exp/{exp_id}.pt')

    # Average over all runs
    train_losses = defaultdict(list)
    test_errors = defaultdict(list)
    for i in range(num_epoch):
        for opt in optimizer_list:
            exp_id = f'{opt}_{wd_group}'
            train_loss = [logs_dict[f'{opt}_{wd_group}_{j}']['train_loss'][i] for j in range(num_exp)]
            train_losses[exp_id + '_mean'].append(np.mean(train_loss))
            train_losses[exp_id + '_max'].append(max(train_loss))
            train_losses[exp_id + '_min'].append(min(train_loss))
            test_error = [logs_dict[f'{opt}_{wd_group}_{j}']['test_error'][i] for j in range(num_exp)]
            test_errors[exp_id + '_mean'].append(np.mean(test_error))
            test_errors[exp_id + '_max'].append(max(test_error))
            test_errors[exp_id + '_min'].append(min(test_error))

    # Plot figure
    i, j = wd // 3, wd % 3
    ax[i, j].plot(range(100), test_errors[f'adam_{wd_group}_mean'], color='orange', label='adam')
    ax[i, j].fill_between(range(100), test_errors[f'adam_{wd_group}_min'], test_errors[f'adam_{wd_group}_max'], color='orange', alpha=0.1)
    ax[i, j].plot(range(100), test_errors[f'adamw_{wd_group}_mean'], color='blue', label='adamw')
    ax[i, j].fill_between(range(100), test_errors[f'adamw_{wd_group}_min'], test_errors[f'adamw_{wd_group}_max'], color='blue', alpha=0.1)
    ax[i, j].plot(range(100), test_errors[f'adamw_fwd_{wd_group}_mean'], color='green', label='adamw_fwd')
    ax[i, j].fill_between(range(100), test_errors[f'adamw_fwd_{wd_group}_min'], test_errors[f'adamw_fwd_{wd_group}_max'], color='green', alpha=0.1)
    ax[i, j].set_ylim(0.2, 0.8)
    ax[i, j].legend()
    ax[i, j].set_xlabel('epochs')
    ax[i, j].set_ylabel('test error')
    ax[i, j].set_title(f'weight decay = {1 * (10 ** -(wd+1)):.6f}')
    ax[i, j].grid(visible=True)

plt.savefig(f'./formulation.png')