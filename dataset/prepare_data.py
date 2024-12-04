import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Define the Dataset class
class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        num_objs = len(coco_annotation)
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        img_id = torch.tensor([img_id])

        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        my_annotation = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


# Define transformations
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# Define a function to create a DataLoader
def create_dataloader(root, annotation_file, batch_size=4, shuffle=True, num_workers=4):
    transform = get_transform()
    dataset = myOwnDataset(root=root, annotation=annotation_file, transforms=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataloader


# Visualize the DataLoader output
def visualize_dataloader(dataloader):
    for images, annotations in dataloader:
        img = images[0].permute(1, 2, 0).numpy()
        boxes = annotations[0]["boxes"].numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for box in boxes:
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

        plt.show()
        break


if __name__ == "__main__":
    root_dir = r"C:\Users\MikeBurgess\PycharmProjects\PythonProject\dataset\data\test"
    annotation_file = r"C:\Users\MikeBurgess\PycharmProjects\PythonProject\dataset\data\test\annotations.json"
    dataloader = create_dataloader(root_dir, annotation_file, batch_size=4)
    visualize_dataloader(dataloader)
