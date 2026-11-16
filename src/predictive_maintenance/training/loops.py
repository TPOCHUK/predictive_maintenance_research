import torch


def train_loop(dataloader, model, loss_fn, optimizer, device):
    """Train for one epoch and return mean MSE."""
    model.train()
    total, n = 0.0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / n


def eval_loop(dataloader, model, loss_fn, device):
    """Check the model and return MSE, MAE, and RMSE."""
    model.eval()
    total_loss = total_abs = total_sq = 0.0
    n = 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            total_loss += loss_fn(y_pred, y).item() * X.size(0)
            total_abs += (y_pred - y).abs().sum().item()
            total_sq += (y_pred - y).square().sum().item()
            n += X.size(0)
    return {"loss": total_loss / n, "mae": total_abs / n, "rmse": (total_sq / n) ** 0.5}
