from dataset.prepare_data import create_dataloader
from training.train import train
from models.model_builder import build_model
from configpy import load_config


def main():
    # Load configuration
    config = load_config("config/config.yaml")

    # Dataloaders
    train_loader = create_dataloader(
        root=config["data"]["train"]["root_dir"],
        annotation_file=config["data"]["train"]["annotations"],
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = create_dataloader(
        root=config["data"]["val"]["root_dir"],
        annotation_file=config["data"]["val"]["annotations"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )


    # Build model and optimizer
    model, optimizer = build_model(config, config["training"]["learning_rate"])

    # Train the model
    train(config, model, train_loader, val_loader, optimizer)
#
if __name__ == "__main__":
    main()
