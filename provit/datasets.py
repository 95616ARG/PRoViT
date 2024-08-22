import torch
import torchvision
from torchvision import transforms as trn
from PRoViT import imagenet
import random
import pathlib
import numpy as np

"""
This file contains functions to get repair sets and generalization sets according to the
specific metrics we wish to measure. n is the number of labels to include in the repair
datasets.

Metric 1 (Experiment 4):
The repair set contains 5 images from each corruption {fog, brightness, frost, snow} and
each severity {1, 2, 3, 4, 5} from the first n ImageNet labels. The generalization sets contain
the remaining 45 images from each corruption and each severity from the same n selected labels.

Metric 2 (Experiment 2):
Select n random labels. The repair set contains 5 images with corruptions fog and
brightness with all severities. This is a total of 50 images per label. The generalization
set contains all other fog and brightness images in that label. 

Metric 3 (Experiments 1 and 3):
The repair set contains 5 images from each corruption {fog, brightness, frost, snow} and
each severity {1, 2, 3, 4, 5} from the n random labels. These images are then augmented 
with 5 different rotations. The degrees of rotation are -10, -5, 0, 5, and 10.
The generalization sets contain
the remaining 45 images from each corruption and each severity from the same n selected labels.
These images are augmented with the same rotations as the repair set.

Metric 4 (Experiment 5):
The repair set contains 5 images from each corruption {fog, brightness, frost, snow} with
severity level 3 from n random labels. The generalization sets contain
the remaining 45 images from each corruption with severity level 3 from the same n selected labels.
"""

def rotate(image, degree, scale=1.):
    _shape = image.shape
    return torchvision.transforms.functional.affine(
            image.reshape(-1, 1, 28, 28),
            angle=degree,
            translate=(0, 0),
            scale=scale,
            shear=0,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        ).reshape(_shape)

def to_rotate_transform(degree):
    return trn.Compose([
        trn.Resize(256, interpolation=trn.InterpolationMode.BILINEAR),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        trn.Lambda(lambda x: rotate(x, degree=degree))
    ])

def process_metric_1(repair_set_params):
    n = repair_set_params.n
    indices = []
    for i in range(repair_set_params.n):
        indices.extend(list(range(n*i, (n*i)+5)))
    repair_sets = []
    for corruption in ('fog', 'brightness', 'frost', 'snow'):
        for severity in range(1, 6):
            repair_sets.append(imagenet.datasets.ImageNet_C((pathlib.Path(repair_set_params.path) / 'imagenet-c').as_posix(), corruption=corruption, severity=severity)\
                    .to(dtype=repair_set_params.dtype, device=repair_set_params.device).subset(indices))
    return repair_sets


def process_metric_2(repair_set_params, random_labels):
    indices = []
    for label in random_labels:
        indices.extend(list(range(label*50, (label*50)+5)))
    repair_sets = []
    for corruption in ('fog', 'brightness'):
        for severity in range(1, 6):
            repair_sets.append(imagenet.datasets.ImageNet_C((pathlib.Path(repair_set_params.path) / 'imagenet-c').as_posix(), corruption=corruption, severity=severity)\
                    .to(dtype=repair_set_params.dtype, device=repair_set_params.device).subset(indices))
    return repair_sets

def process_metric_3(repair_set_params, random_labels):
    # Take the first 5 base images per label.
    indices = []
    for label in random_labels:
        indices.extend(list(range(label*50, (label*50)+5)))

    # Rotate images by -10, -5, 0, 5, and 10 degrees.
    degrees = np.arange(-10., 11., 5.)
    return [
        imagenet.datasets.ImageNet_C(
            (pathlib.Path(repair_set_params.path) / 'imagenet-c').as_posix(),
            corruption=corruption,
            severity=severity,
            transform=to_rotate_transform(degree)
        ).to(dtype=repair_set_params.dtype, device=repair_set_params.device).subset(indices)
        for degree in degrees
        for severity in range(1, 6)
        for corruption in ('fog','brightness','frost','snow')
    ]


def process_metric_4(repair_set_params, random_labels):
    indices = []
    for label in random_labels:
        indices.extend(list(range(label*50, (label*50)+5)))
    repair_sets = []
    for corruption in ('fog', 'brightness', 'frost', 'snow'):
        repair_sets.append(imagenet.datasets.ImageNet_C((pathlib.Path(repair_set_params.path) / 'imagenet-c').as_posix(), corruption=corruption, severity=3)\
                .to(dtype=repair_set_params.dtype, device=repair_set_params.device).subset(indices))
    return repair_sets


class GetRepairSetParameters:
    def __init__(self, n, metric, dtype, path, device, seed):
        self.n = n
        self.metric = metric
        self.dtype = dtype
        self.path = path
        self.device = device
        self.seed = seed

def get_repair_sets(n, metric=1, dtype=torch.float32, path='./data', device='cpu', seed=0):
    """Return the repair sets according to the {metric} and designated number of labels {n}.
    Repair sets are returned as a list of Datasets.
    """
    # create parameter object
    repair_set_params = GetRepairSetParameters(n, metric, dtype, path, device, seed)

    functions_for_metrics = {
        1: process_metric_1,
    }

    functions_for_metrics_using_random_labels = {
        2: process_metric_2,
        3: process_metric_3,
        4: process_metric_4,
    }

    if metric in functions_for_metrics:
        return functions_for_metrics[metric](repair_set_params)

    elif metric in functions_for_metrics_using_random_labels:
        np.random.seed(seed)
        random.seed(seed)
        random_labels = np.asarray(list(range(0, 999)))
        random.shuffle(random_labels)
        random_labels = random_labels[:n]
        print("Random labels: ", random_labels)

        return functions_for_metrics_using_random_labels[metric](repair_set_params, random_labels)
    else:
        raise NotImplementedError(f"Metric {repair_set_params.metric} is not implemented for repair sets.")


def get_gen_set_metric_1(gen_set_params, random_labels):
    n = len(random_labels)
    indices = []
    for i in range(n):
        indices.extend(list(range((n*i)+5, (n*(i+1)+49))))

    return {
                corruption : {
                        severity : imagenet.datasets.ImageNet_C((pathlib.Path(gen_set_params.path) / 'imagenet-c').as_posix(), corruption=corruption, severity=severity)\
                                            .to(dtype=gen_set_params.dtype, device=gen_set_params.device).subset(indices)
                        for severity in range(1, 6)
                } for corruption in ('fog','brightness','frost','snow')
            }

def get_gen_set_metric_2(gen_set_params, random_labels):
    indices = []
    for label in random_labels:
        indices.extend(list(range((label*50)+5, (label+1)*50)))
    return {
        corruption : {
                severity : imagenet.datasets.ImageNet_C((pathlib.Path(gen_set_params.path) / 'imagenet-c').as_posix(), corruption=corruption, severity=severity)\
                                    .to(dtype=gen_set_params.dtype, device=gen_set_params.device).subset(indices)
                for severity in range(1, 6)
        } for corruption in ('fog','brightness')
            }

def get_gen_set_metric_3(gen_set_params, random_labels):
    # Take the remaining 45 base images per label.
    indices = []
    for label in random_labels:
        indices.extend(list(range((label*50)+5, (label+1)*50)))

    # Rotate images by -10, -5, 0, 5, and 10 degrees.
    degrees = np.arange(-10., 11., 5.)
    return [
        imagenet.datasets.ImageNet_C(
            (pathlib.Path(gen_set_params.path) / 'imagenet-c').as_posix(),
            corruption=corruption,
            severity=severity,
            transform=to_rotate_transform(degree)
        ).to(dtype=gen_set_params.dtype, device=gen_set_params.device).subset(indices)
        for degree in degrees
        for severity in range(1, 6)
        for corruption in ('fog','brightness','frost','snow')
    ]


def get_gen_set_metric_4(gen_set_params, random_labels):
    indices = []
    for label in random_labels:
        indices.extend(list(range((label*50)+5, (label+1)*50)))

    return {
                corruption : {
                        severity : imagenet.datasets.ImageNet_C((pathlib.Path(gen_set_params.path) / 'imagenet-c').as_posix(), corruption=corruption, severity=severity)\
                                            .to(dtype=gen_set_params.dtype, device=gen_set_params.device).subset(indices)
                        for severity in [3]
                } for corruption in ('fog','brightness','frost','snow')
            }


def get_gen_sets(n, metric=1, dtype=torch.float32, path='./data', device='cpu', seed=0):
    """Return the generalization sets according to the {metric} and designated number of labels {n}.
    Generalization sets are returned as a dict of dicts of Datasets.
    """
    gen_set_params = GetRepairSetParameters(n, metric, dtype, path, device, seed)

    functions_for_gen_sets_using_random_labels = {
        1: get_gen_set_metric_1,
        2: get_gen_set_metric_2,
        3: get_gen_set_metric_3,
        4: get_gen_set_metric_4,
    }

    if metric in functions_for_gen_sets_using_random_labels:
        np.random.seed(seed)
        random.seed(seed)
        random_labels = np.asarray(list(range(0, 999)))
        random.shuffle(random_labels)
        random_labels = random_labels[:n]
        return functions_for_gen_sets_using_random_labels[metric](gen_set_params, random_labels)
    else:
        raise NotImplementedError(f"Metric {gen_set_params.metric} is not implemented for gen sets.")


def get_drawdown_set(dtype=torch.float32, path='./data', device='cpu'):
    """Return the ILSVRC2012 validation set, used for computing drawdown."""

    return imagenet.datasets.ImageNet(root=(pathlib.Path(path) / 'ILSVRC2012').as_posix()).to(dtype=dtype, device=device)
