import numpy as np
import torch


def train_step(model, x, targets, optimizer, loss_fn):
    M = model.M

    out = model(x, targets)

    loss = loss_fn(out, targets)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    level_losses = np.zeros(M)
    with torch.no_grad():
        for m in range(M):
            level_losses[m] = loss_fn(out[:, :, m], targets[:, m]).item()

    return {'loss': loss.item(), 'level_losses': level_losses}


def val_step(model, x, targets, loss_fn):
    M = model.M

    out = model(x, targets, tf_with_eval=True)
    val_out = out.clone()
    loss = loss_fn(val_out, targets).item()

    level_losses = np.zeros(M)
    for m in range(M):
        level_losses[m] = loss_fn(val_out[:, :, m], targets[:, m]).item()

    return {'loss': loss, 'level_losses': level_losses}
