from dataset.prepare_data import get_dataloader
from training.train import train
from models.model_builder import build_model
from config import load_config

def main():
    config = load_config("config/config.yaml")

    # Prepare dataloaders
    train_loader = get_dataloader(config, train=True)
    val_loader = get_dataloader(config, train=False)

    # Build model and optimizer
    model, optimizer = build_model(config, config["training"]["learning_rate"])

    # Train the model
    train(config, model, train_loader, val_loader, optimizer)

if __name__ == "__main__":
    main()
