import timm
import torch
import os
def get_arch(arch, dataset, saved_weight = "", pretrained=False, resume=False):
    if dataset == 'imagenet':
        return timm.create_model(model_name=arch, num_classes=1000, pretrained=pretrained)
    elif dataset == 'imagenette' or dataset == 'imagenet-10':
        # return timm.create_model(model_name=arch, num_classes=10, pretrained=pretrained)
        # file_path = os.path.join('/private/workpace/ensrobusttraining/imagenette_logs/check_points',arch,'pytorch_model.bin')
        return timm.create_model(model_name=arch, num_classes=10, pretrained=pretrained,pretrained_cfg_overlay=dict(file=saved_weight))
    elif dataset=='tiny-imagenet':
        return timm.create_model(model_name=arch, num_classes=200, pretrained=pretrained)
    # elif dataset == 'cifar10' or dataset=='cifar100' or dataset == 'GTSRB':
    #     if dataset == 'cifar10':
    #         num_class = 10
    #     elif dataset == 'cifar100':
    #         num_class = 100
    #     else:
    #         num_class = 43
    #     if not resume:
    #         import models
    #     else:
    #         from .. import models
    #     return getattr(models, arch)(num_class)

    elif dataset == 'cifar10-vit':
        if arch == 'ViT':
            model = timm.create_model(model_name='vit_base_patch16_224',  pretrained=pretrained)
            model.head = torch.nn.Linear(model.head.in_features, 10)
            return model
        elif arch == 'ViT-sam':
            model = timm.create_model(model_name='vit_base_patch16_224_sam', pretrained=pretrained)
            model.head = torch.nn.Linear(model.head.in_features, 10)
            return model
    else:
        raise NotImplementedError('architecture {} is not supported in dataset'.format(arch, dataset))

def get_normalize(dataset):
    class Normalize(torch.nn.Module):

        def __init__(self, mean, std):
            super(Normalize, self).__init__()
            self.mean = mean
            self.std = std

        def forward(self, input):
            size = input.size()
            x = input.clone()
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
            return x

    if dataset == 'cifar10':
        mean = torch.Tensor([0.4914, 0.4822, 0.4465])
        std = torch.Tensor([0.2471, 0.2435, 0.2616])
    elif dataset == 'tiny-imagenet':
        mean = torch.Tensor([0.4802, 0.4481, 0.3975])
        std = torch.Tensor([0.2770, 0.2691, 0.2821])
    elif dataset == 'cifar100':
        mean = torch.Tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
        std = torch.Tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    elif dataset == 'cifar10-vit':
        mean = torch.Tensor([0.4914, 0.4822, 0.4465])
        std = torch.Tensor([0.2471, 0.2435, 0.2616])
    elif dataset == 'imagenet' or dataset == 'imagenette' or dataset == 'imagenet-10':
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
    elif dataset == 'GTSRB':
        mean = torch.Tensor([0, 0, 0])
        std = torch.Tensor([1, 1, 1])
    else:
        raise NotImplementedError

    return Normalize(mean=mean, std=std)

class WrapModel(torch.nn.Module):
    def __init__(self,  model, normalizer):
        super(WrapModel, self).__init__()
        self.model = model
        self.normal = normalizer

    def forward(self, input):
        return self.model(self.normal(input))
    def features(self, input):
        return self.model.features(self.normal(input))
    def eval(self):
        self.model.eval()