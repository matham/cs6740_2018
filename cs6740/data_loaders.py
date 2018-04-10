import torchvision.datasets as dset
import random


class CocoDataset(dset.CocoCaptions):

    def __init__(self, *largs, **kwargs):
        super(CocoDataset, self).__init__(*largs, **kwargs)
        n = len(self.ids)
        # for every example, create a negative example and store them i >= len(ids)
        mapping = self.data_mapping = list(range(2 * n))

        for i in range(n):
            val = i
            while val == i:
                val = random.randint(0, n - 1)
            mapping[n + i] = val

    def __getitem__(self, index):
        img, captions = super(CocoDataset, self).__getitem__(index)
        caption = captions[random.randint(0, len(captions) - 1)]
        label = 1 if index < len(self.data_mapping) else 0
        return img, caption, label

    def __len__(self):
        return len(self.data_mapping)
