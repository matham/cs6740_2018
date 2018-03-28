import torchvision.datasets as dset
import random


class CocoDataset(dset.CocoCaptions):

    def __getitem__(self, item):
        img, captions = item
        return img, captions[random.randint(0, len(captions) - 1)]