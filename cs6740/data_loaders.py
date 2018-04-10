import torchvision.datasets as dset
import random


class CocoDataset(dset.CocoCaptions):

    def __init__(self, *largs, **kwargs):
        super(CocoDataset, self).__init__(*largs, **kwargs)
        self.orignal_n = n = len(self.ids)
        # for every example, create a negative example and store them i >= len(ids)
        mapping = self.data_mapping = list(range(2 * n))

        for i in range(n):
            val = i
            while val == i:
                val = random.randint(0, n - 1)
            mapping[n + i] = val

    def __getitem__(self, index):
        if index < len(self.ids):
            img, caption = super(CocoDataset, self).__getitem__(index)
            label = 1.
        else:
            _, caption = super(CocoDataset, self).__getitem__(index - len(self.ids))
            img, _ = super(CocoDataset, self).__getitem__(self.data_mapping[index])
            label = -1.
        return img, caption, label

    def __len__(self):
        return len(self.data_mapping)
