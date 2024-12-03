from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch

def build_model(config, lr):
    # Load Faster R-CNN with ResNet-50 backbone
    model = fasterrcnn_resnet50_fpn(
        num_classes=config["model"]["num_classes"],
        pretrained=True  # Start with ImageNet weights
    )

    for param in model.backbone.body.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    return model, optimizer
