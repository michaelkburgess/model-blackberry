import torch

def move_to_device(batch, device):
    images, targets = batch

    # Debugging the structure of images and targets
    # print(f"Images: {images}")
    # print(f"Targets: {targets}")

    images = [image.to(device) for image in images]

    if not isinstance(targets, list) or not all(isinstance(t, dict) for t in targets):
        raise ValueError("Targets must be a list of dictionaries with tensor values.")

    targets = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
        for target in targets
    ]
    return images, targets