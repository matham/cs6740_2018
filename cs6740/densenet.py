import torchvision.models as tv_models
import torch.nn as nn


def DenseNet121(*largs, num_output_features, **kwargs):
    net = tv_models.densenet121(*largs, **kwargs)
    net.classifier = nn.Linear(net.classifier.in_features, num_output_features)
    return net
