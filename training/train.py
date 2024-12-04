import torch
from training.evaluation import evaluate
from utils.helper import move_to_device
import os

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move batch to device
        images, targets = move_to_device(batch, device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        try:
            loss_dict = model(images, targets)
            print(f"Loss dict: {loss_dict}")  # Print loss dictionary for debugging
            loss = sum(loss for loss in loss_dict.values())
            print(f"Loss: {loss.item()}")
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
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

        # Training phase
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        avg_val_loss = evaluate(model, val_loader, device)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save the best model
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
