import torch

from torchvision.datasets import CocoCaptions as Coco

coco_dataset = Coco(root="val2017", annFile="annotations/captions_val2017.json")

for item in coco_dataset:
    print(item)
