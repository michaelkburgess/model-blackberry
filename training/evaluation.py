import torch

def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            # Forward pass
            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())

            # Accumulate loss
            running_loss += loss.item()

    # Calculate average validation loss
    avg_val_loss = running_loss / len(dataloader)
    return avg_val_loss
