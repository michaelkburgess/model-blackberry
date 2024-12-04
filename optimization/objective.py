import torch
from training.train import train_one_epoch
from training.evaluation import evaluate
from models.model_builder import build_model

def objective(trial, train_loader, val_loader, config):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # Update config with suggested hyperparameters
    config["training"]["batch_size"] = batch_size
    config["training"]["learning_rate"] = learning_rate

    # Build the model and optimizer
    model, optimizer = build_model(config, learning_rate)

    # Move model to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train and evaluate for a subset of epochs
    epochs = config["training"].get("trial_epochs", 5)  # Use fewer epochs for trial
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, device)

    # Evaluate and return validation loss
    val_loss = evaluate(model, val_loader, device)
    return val_loss
