import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_V2_Weights

# Load the model
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_V2_Weights)

# Print the model architecture
print(model)