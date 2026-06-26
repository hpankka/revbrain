from utils.data_utils import mu_law_quantize


def _preprocess_targets(y):
    y = y / 100
    return mu_law_quantize(y)


def train_step(model, x, targets, optimizer, loss_fn):
    y = _preprocess_targets(targets)

    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {'loss': loss.item()}


def val_step(model, x, targets, loss_fn):
    y = _preprocess_targets(targets)

    out = model(x, return_logits_with_eval=True)
    loss = loss_fn(out, y).item()

    return {'loss': loss}
