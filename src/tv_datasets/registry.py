import sys
import inspect
import random
import torch
import copy

from torch.utils.data.dataset import random_split

from src.tv_datasets.cars import Cars
from src.tv_datasets.cifar10 import CIFAR10
from src.tv_datasets.cifar100 import CIFAR100
from src.tv_datasets.dtd import DTD
from src.tv_datasets.emnist import EMNIST
from src.tv_datasets.eurosat import EuroSAT, EuroSATVal
from src.tv_datasets.fashionmnist import FashionMNIST
from src.tv_datasets.fer2013 import FER2013
from src.tv_datasets.flowers102 import Flowers102
from src.tv_datasets.food101 import Food101
from src.tv_datasets.gtsrb import GTSRB
from src.tv_datasets.imagenet import ImageNet
from src.tv_datasets.kmnist import KMNIST
from src.tv_datasets.mnist import MNIST
from src.tv_datasets.oxfordpets import OxfordIIITPet
from src.tv_datasets.pcam import PCAM
from src.tv_datasets.resisc45 import RESISC45
from src.tv_datasets.sst2 import RenderedSST2
from src.tv_datasets.stl10 import STL10
from src.tv_datasets.sun397 import SUN397
from src.tv_datasets.svhn import SVHN

from src.tv_datasets.vegetables   import Vegetables
from src.tv_datasets.kvasir       import Kvasir
from src.tv_datasets.intelimages  import IntelImages
from src.tv_datasets.weather      import Weather
from src.tv_datasets.catsdogs     import CatsDogs
from src.tv_datasets.landscape    import Landscape
from src.tv_datasets.garbage      import Garbage
from src.tv_datasets.fruits360 import Fruits360
from src.tv_datasets.mangoleafbd import MangoLeafBD
from src.tv_datasets.beans import Beans


registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    assert val_fraction > 0. and val_fraction < 1.
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.train_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )
    if new_dataset_class_name == 'MNISTVal':
        assert trainset.indices[0] == 36044


    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=16, val_fraction=0.1, max_val_samples=5000):
    if dataset_name.endswith('Val'):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split('Val')[0]
            base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
            dataset = split_train_into_train_val(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
            return dataset
    else:
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        dataset_class = registry[dataset_name]
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers
    )
    return dataset
