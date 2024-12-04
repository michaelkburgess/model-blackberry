from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torch


def build_model(config_path, lr):

    # Load Faster R-CNN with ResNet-50 backbone
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # Use the default pre-trained weights
    )

    # Modify the number of classes to match your dataset
    num_classes =4
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Modify the box predictor for the new number of classes
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, num_classes)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, num_classes * 4)

    # Set up the optimizer (only for the layers that require gradients)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    return model, optimizer
