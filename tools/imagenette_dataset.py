import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
class Dataset(torchvision.datasets.DatasetFolder):
    def __init__(self, dataset):
        assert isinstance(dataset, torchvision.datasets.DatasetFolder)
        self.loader    = dataset.loader
        self.classes   = dataset.classes
        self.samples   = dataset.samples
        self.y         = dataset.targets
        self.transform = dataset.transform
        self.target_transform = None

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def __len__(self):
        return len(self.y)
def datasetImageNett(root='./data', train=True, transform=None):
    if train: root = os.path.join(root, 'train')
    else: root = os.path.join(root, 'val')
    return torchvision.datasets.ImageFolder(root=root, transform=transform)
def get_transforms(dataset, train=True):
    assert (dataset == 'imagenet' or dataset == 'imagenette'or dataset == 'imagenet-10' )
    # mean = torch.Tensor([0.485, 0.456, 0.406])
    # std = torch.Tensor([0.229, 0.224, 0.225])
    if train:
        comp = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ]
    else:
        comp = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ]


    trans = transforms.Compose( comp )

    return trans

def get_dataset(dataset, root='./data', train=True):
    assert (dataset == 'imagenet' or dataset == 'imagenet-mini' or dataset == 'imagenette' or dataset =='tiny-imagenet' or dataset == 'imagenet-10')
    if dataset == 'imagenet' or dataset == 'imagenet-mini' or dataset == 'imagenette'  or dataset == 'imagenet-10':
        transform = get_transforms(dataset, train=train)
    # elif  dataset =='tiny-imagenet':
    #     transform = get_tiny_transforms(dataset, train=train)
    if dataset == 'imagenette':
        target_set = datasetImageNett(root=root, train=train, transform=transform)
    # if dataset == 'imagenet':
    #     target_set = datasetImageNet(root=root, train=train, transform=transform)
    # if dataset == 'imagenet-mini':
    #     target_set = datasetImageNetMini(root=root, train=train, transform=transform)
    # if dataset == 'imagenet-10':
    #     target_set = datasetImageNet10(root=root, train=train, transform=transform)
    # if dataset == 'tiny-imagenet':
    #     target_set = datasetTinyImageNet(root=root,train=train, transform=transform)
    target_set = Dataset(target_set)

    return target_set

def get_image_loader(args):
    train_dataset = get_dataset(root=args.data_dir,dataset=args.dataset,train=True)
    if args.dataset == 'imagenette' or args.dataset == 'imagenet-10':
        num_classes = 10
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200
    else:
        raise NotImplementedError

    test_dataset = get_dataset(root=args.data_dir,dataset=args.dataset,train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)
    return train_loader, test_loader