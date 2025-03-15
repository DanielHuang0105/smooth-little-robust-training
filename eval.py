import torch
from torchvision import transforms
from tensorboardX import SummaryWriter
import os
import pandas as pd
import tools.imagenette_dataset as imagenette_dataset
import tools.imagenette_model as imagenette_model
from transferability_eval import tranferability_validate,tranferability_validate_v2
from tools.model import generate_model
from argparse import ArgumentParser
from pathlib import Path
from cosine_simlaratiy_eval import compute_cos
from robustness import datasets
import train_arg
if __name__ == "__main__":
    torch.manual_seed(1)
    parser = ArgumentParser()

    parser = train_arg.add_args_to_parser(train_arg.EVAL_ATTACK_ARGS, parser)
    parser = train_arg.add_args_to_parser(train_arg.TRAINING_ARGS, parser)
    parser.add_argument("--model_weight_path", type=str, default="")
    parser.add_argument("--saved_excel_path", type=str, default="")
    parser.add_argument("--method_name", type=str, default="")
    parser.add_argument("--is_ours", type=str, default="ours")
    args = parser.parse_args()

    assert(args.saved_excel_path!='')
    saved_excel_path = Path(args.saved_excel_path)
    saved_excel_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter('./logs/tensorboard/esp0.03')
    bs = args.batch_size
    if args.dataset == 'cifar':
        # path to cifar10
        dataset = datasets.CIFAR('./dataset/cifar10')
        train_loader, val_loader = dataset.make_loaders(2, bs, data_aug=False)

        if args.is_ours == "others":
            # model tranied by package robustness
            model = generate_model("resnet18",dataset,"./logs/checkpoints/resnet18/checkpoint.pt.best").cuda()
        elif args.is_ours == "littlerobust":
            # model tranied by package robustness
            model = generate_model("resnet18",dataset,"./logs/checkpoints/resnet18_l2_0.03/76a75902-fb32-4ec5-96ca-a0f064591095/checkpoint.pt.best").cuda()
        elif args.is_ours == "ours" and args.model_weight_path!='':
            model = generate_model("resnet18",dataset,"./logs/checkpoints/resnet18/checkpoint.pt.best").cuda()
            state_dict = torch.load(args.model_weight_path)
            model.load_state_dict(state_dict)
        
        # target model using to eval transferability
        target_model_list = []

        # model tranied by package robustness
        if args.defense_type == 'at':
            # target_model_list.append(generate_model("resnet50",dataset,"./logs/checkpoints/resnet50/7beb7e0a-fe7d-4c63-adf6-73625f64bad4/checkpoint.pt.best").cuda())    
            target_model_list.append(generate_model("vgg16",dataset,"./logs/checkpoints/vgg16/362d4020-80b0-464b-bce7-87247a0bb447/checkpoint.pt.best").cuda())
            # target_model_list.append(generate_model("densenet121",dataset,"./logs/checkpoints/densenet121/7a1f3d03-6e3a-466c-9519-4973a9cc3dec/checkpoint.pt.best").cuda())
            # target_model_list.append(generate_model("inceptionv3",dataset,"./logs/checkpoints/inceptionv3/83ed83f3-7df3-4524-bb0b-b08fb2fc7c04/checkpoint.pt.best").cuda())
        else:
            target_model_list.append(generate_model("resnet50",dataset,"./logs/checkpoints/resnet50/7beb7e0a-fe7d-4c63-adf6-73625f64bad4/checkpoint.pt.best").cuda())    
            target_model_list.append(generate_model("vgg16",dataset,"./logs/checkpoints/vgg16/362d4020-80b0-464b-bce7-87247a0bb447/checkpoint.pt.best").cuda())
            target_model_list.append(generate_model("densenet121",dataset,"./logs/checkpoints/densenet121/7a1f3d03-6e3a-466c-9519-4973a9cc3dec/checkpoint.pt.best").cuda())
            target_model_list.append(generate_model("inceptionv3",dataset,"./logs/checkpoints/inceptionv3/83ed83f3-7df3-4524-bb0b-b08fb2fc7c04/checkpoint.pt.best").cuda())
    
    elif args.dataset == 'imagenette':
        train_loader, val_loader = imagenette_dataset.get_image_loader(args) #todo
        model = imagenette_model.get_arch('resnet50', args.dataset,pretrained=False)
        normalize = imagenette_model.get_normalize(args.dataset)
        model = imagenette_model.WrapModel(model=model, normalizer=normalize)
        if args.is_ours == "ours":
            weights = torch.load("./imagenette_logs/saved_model/imagenette-resnet50-latest.pth", map_location=torch.device('cpu'))
        elif args.is_ours == "others":
            weights = torch.load("./imagenette_logs/check_points/imagenette_resnet50/pytorch_model.bin", map_location=torch.device('cpu'))
        elif args.is_ours == "littlerobust":
            weights = torch.load("./imagenette_logs/check_points/imagenette_resnet50_l2_1/imagenette-resnet50-best.pth", map_location=torch.device('cpu'))

        model.load_state_dict(weights,strict=False)      
        model = model.to("cuda") 
        target_model_list = []
        if args.defence_type == 'at':
            target_model_weight_dict = {"vgg16_bn":"./imagenette_logs/check_points/imagenette_vgg16_bn_linf_4/pytorch_model.bin"}
        else :
            target_model_weight_dict = {"vgg16_bn":"./imagenette_logs/check_points/imagenette_vgg16_bn/pytorch_model.bin",
                                        "mobilenetv2_100":"./imagenette_logs/check_points/imagenette_mobilenetv2_100/pytorch_model.bin",
                                        "xception41":"./imagenette_logs/check_points/imagenette_xception41/pytorch_model.bin",
                                        "inception_resnet_v2":"./imagenette_logs/check_points/imagenette_inception_resnet_v2/pytorch_model.bin",
                                        }

        for (key,value) in target_model_weight_dict.items():
            target_model = imagenette_model.WrapModel(model=imagenette_model.get_arch(key, args.dataset,pretrained=False), normalizer=imagenette_model.get_normalize(args.dataset))
            weights = torch.load(value, map_location=torch.device('cpu'))
            target_model.load_state_dict(weights,strict=False)
            target_model = target_model.to("cuda")
            target_model_list.append(target_model)
    metrix = tranferability_validate_v2(args, val_loader, target_model_list, model, device, transforms.ToTensor(), 0, writer)



    # save data
    data = {}
    if args.dataset == 'cifar':
        data.update({"model name":"resnet18"+args.method_name})
    elif args.dataset == 'imagenette':
        data.update({"model name":"resnet50"+args.method_name})
    transferability_path = os.path.join(saved_excel_path,"transferability.xlsx")
    for idx in range(len(metrix)):
        data.update({f"model{idx}":str(round(metrix[idx].attack_rate * 100, 2))})
    
    df_data_save = pd.DataFrame(data,index=[0])
    if not os.path.exists(transferability_path):
        df_data_save.to_excel(transferability_path, 'sheet1', index=False)
    else:
        # read original excel data first, then write into excel
        original_data = pd.read_excel(transferability_path)
        save_data = pd.concat([original_data,df_data_save])
        save_data.to_excel(transferability_path, index=False)







