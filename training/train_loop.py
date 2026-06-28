import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.model_utils import RVQ_MODELS
from utils.training_utils import get_train_steps, save_and_plot_losses


def train_loop(model, data_loaders, config, dir_):
    """
    Train a model and save checkpoints, losses, and loss plots to dir_.
    Checkpoints and losses are saved every 10 epochs.

    Per-batch step logic is dispatched based on model type (RVQ or WaveNet).

    Params:
    model: nn.Module, the model to train
    data_loaders: (train_loader, val_loader) tuple
    config: training config dict with keys 'model', 'device', 'model_config',
            and 'train_config' (epochs, lr, batch_size)
    dir_: output directory
    """

    # Set up constants
    is_rvq = config['model'] in RVQ_MODELS
    M = config['model_config'].get('M')
    epochs = config['train_config']['epochs']
    device = torch.device(config['device'])

    # Set up data loaders and training steps
    train_loader, val_loader = data_loaders
    train_step, val_step = get_train_steps(is_rvq)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['train_config']['lr'], eps=0.001)
    scheduler = None if is_rvq else optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    loss_fn = nn.CrossEntropyLoss()

    # Set up loss and checkpoint dirs
    loss_path, cp_path = os.path.join(dir_, 'Losses'), os.path.join(dir_, 'Checkpoints')
    os.makedirs(loss_path, exist_ok=True)
    os.makedirs(cp_path, exist_ok=True)
    
    # Initialize losses
    train_loss, val_loss = [], []
    train_losses_levels = np.zeros((M, epochs)) if is_rvq else None
    val_losses_levels   = np.zeros((M, epochs)) if is_rvq else None

    for i in range(epochs):

        # Train
        model.train()
        epoch_train_loss = 0.0
        epoch_train_ll = np.zeros(M) if is_rvq else None

        for batch_idx, (x, targets) in enumerate(train_loader):
            x, targets = x.to(device), targets.to(device)
            result = train_step(model, x, targets, optimizer, loss_fn)
            epoch_train_loss += result['loss']
            if is_rvq:
                epoch_train_ll += result['level_losses']

        if scheduler:
            scheduler.step()

        train_loss.append(epoch_train_loss / (batch_idx + 1))
        if is_rvq:
            train_losses_levels[:, i] = epoch_train_ll / (batch_idx + 1)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_ll = np.zeros(M) if is_rvq else None

        with torch.no_grad():
            for batch_idx, (x, targets) in enumerate(val_loader):
                x, targets = x.to(device), targets.to(device)
                result = val_step(model, x, targets, loss_fn)
                epoch_val_loss += result['loss']
                if is_rvq:
                    epoch_val_ll += result['level_losses']

        val_loss.append(epoch_val_loss / (batch_idx + 1))
        if is_rvq:
            val_losses_levels[:, i] = epoch_val_ll / (batch_idx + 1)

        print(f"Epoch {i + 1}, loss: {train_loss[-1]:.4f}, val loss: {val_loss[-1]:.4f}")

        if (i + 1) % 10 == 0:
            cp = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            if scheduler:
                cp['scheduler'] = scheduler.state_dict()
            torch.save(cp, os.path.join(cp_path, f'checkpoint_{i + 1}.tar'))
            save_and_plot_losses((train_loss, val_loss, train_losses_levels, val_losses_levels), loss_path)

    torch.save(model.state_dict(), os.path.join(dir_, f'model.pth'))
    save_and_plot_losses((train_loss, val_loss, train_losses_levels, val_losses_levels), loss_path)
