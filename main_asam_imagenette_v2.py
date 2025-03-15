import os
import numpy as np
from robustness import datasets
from torchvision import transforms
from argparse import ArgumentParser
from torch.optim.lr_scheduler import CosineAnnealingLR
from ens_train import *
from third_party.asam import SAM,ASAM
from transferability_eval import tranferability_validate_v2

from tools.model import generate_model
import tools.imagenette_dataset as imagenette_dataset
import tools.imagenette_model as imagenette_model
from pathlib import Path
import train_arg
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter

def getmodel_torchvision(model_name):
    if model_name == 'vit_l_16':
        wname = 'IMAGENET1K_SWAG_LINEAR_V1'
        model = getattr(torchvision.models,
                                model_name)(weights=wname)
    elif 'inception' in model_name:
        model = torchvision.models.__dict__[model_name](
            weights='DEFAULT', transform_input=False)
    else:
        model = torchvision.models.__dict__[model_name](
            weights='DEFAULT')
    return model

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = train_arg.add_args_to_parser(train_arg.PGD_ARGS, parser)
    parser = train_arg.add_args_to_parser(train_arg.EVAL_ATTACK_ARGS, parser)
    parser = train_arg.add_args_to_parser(train_arg.SAM_ARGS, parser)
    parser = train_arg.add_args_to_parser(train_arg.TRAINING_ARGS, parser)
    # 3: adv train with sam 7: normal train
    parser.add_argument("--train_type",type=[3,7],default=3) 
    parser.add_argument("--random_seeds",type=int,default=1) 
    args = parser.parse_args()
    # torch.manual_seed(seed)
    seed_everything(args.random_seeds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter('./writer')
    epochs = args.epoch
    maximum_asr = 0
    # zoo model 
    zoo_models = []
    # target model using to eval transferability
    target_model_list = []
    if args.dataset == 'cifar':
        dataset = datasets.CIFAR(args.data_dir)
        train_loader, val_loader = dataset.make_loaders(2, args.batch_size, data_aug=False)
        # model to be trained
        model = generate_model("resnet18",dataset,"./logs/checkpoints/resnet18/checkpoint.pt.best").cuda()               
        # target_model_list.append(generate_model("resnet50",dataset,"/private/workpace/ensrobusttraining/logs/checkpoints/resnet50/7beb7e0a-fe7d-4c63-adf6-73625f64bad4/checkpoint.pt.best").cuda())    
        target_model_list.append(generate_model("vgg16",dataset,"./logs/checkpoints/vgg16/362d4020-80b0-464b-bce7-87247a0bb447/checkpoint.pt.best").cuda())

    elif args.dataset == 'imagenette':
        train_loader, val_loader = imagenette_dataset.get_image_loader(args)
        # normal train
        if args.train_type == 7: 
            model = getmodel_torchvision(args.arch)
            model.to(device)
            target_model = getmodel_torchvision("densenet121")
            # target_model_list.append(imagenette_model.WrapModel(model=imagenette_model.get_arch('vgg16_bn', args.dataset, "/private/workpace/ensrobusttraining/imagenette_logs/check_points/vgg16_bn/pytorch_model.bin",pretrained=True), normalizer=imagenette_model.get_normalize(args.dataset)).cuda())
            target_model.to(device)
            target_model_list.append(target_model)
        # our method
        else: 
            model = getmodel_torchvision("resnet50")
            weights = torch.load("./imagenette_logs/check_points_v2_torchvision/imagenette_resnet50/pytorch_model.bin", map_location=torch.device('cpu'))
            keys = model.load_state_dict(weights,strict=False)  
            model = model.to("cuda")      

            target_model = getmodel_torchvision("densenet121")
            weights = torch.load("./imagenette_logs/check_points_v2_torchvision/imagenette_densenet121/pytorch_model.bin", map_location=torch.device('cpu'))
            target_model.load_state_dict(weights,strict=False)
            target_model = target_model.to("cuda")
            target_model_list.append(target_model)
    assert(args.saved_dir!='')
    save_dir = Path(args.saved_dir+args.dataset+'_'+args.arch)
    save_dir.mkdir(parents=True, exist_ok=True)
    

    # optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    if args.train_type == 7:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)   
    else:
        minimizer = eval(args.minimizer)(optimizer, model, rho=args.rho, eta=args.eta)
        scheduler = CosineAnnealingLR(minimizer.optimizer, T_max=epochs)             

    for epoch in range(epochs): 
        if args.train_type == 3:
            # --loss_schema average
            adv_train_withclean_withasam(train_loader, criterion, minimizer, model, zoo_models, writer, epoch, args)
        elif args.train_type == 7:
            normal_train(train_loader, criterion, optimizer, model, zoo_models, writer, epoch, args)
        validate(val_loader, model, criterion, writer, epoch)

        # normal train
        if args.train_type == 7:
            torch.save(
                model.state_dict(),
                save_dir
                / "pytorch_model.bin",
            )
        # our method
        else:
            if (epoch+1)%5 == 0:
                metrix = tranferability_validate_v2(args, val_loader, target_model_list, model, device, transforms.ToTensor(), epoch, writer)
                asr = metrix[0].attack_rate
                if asr>maximum_asr:
                    maximum_asr = asr
                    torch.save(
                        model.state_dict(),
                        save_dir
                        / "{}-resnet50-best.pth".format(args.dataset),
                    )
                torch.save(
                    model.state_dict(),
                    save_dir
                    / "{}-resnet50-latest.pth".format(args.dataset),
                )

        scheduler.step()