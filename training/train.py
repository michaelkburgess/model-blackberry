import torch
from training.evaluation import evaluate
import os

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        # Move images and targets to the appropriate device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss for logging
        running_loss += loss.item()

    # Calculate average loss
    avg_loss = running_loss / len(dataloader)
    return avg_loss


def train(config, model, train_loader, val_loader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config["training"].get("patience", 5)

    save_dir = config["training"].get("save_dir", ".")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")

        # Training Phase
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation Phase
        avg_val_loss = evaluate(model, val_loader, device)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save the model if the validation loss is the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
