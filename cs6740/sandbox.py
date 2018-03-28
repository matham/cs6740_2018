import torch
import torchtext

from torchvision.datasets import CocoCaptions as Coco

coco_dataset = Coco(root="../val2017", annFile="../annotations/captions_val2017.json")

embeddings = torchtext.vocab.GloVe(name="6B", dim=50)
print(type(embeddings))
