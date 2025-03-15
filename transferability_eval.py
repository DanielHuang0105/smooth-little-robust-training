import torch

from tqdm import tqdm
from prettytable import PrettyTable
from defence.jpeg_compression import Jpeg_compression
from defence.bit_depth_reduction import BitDepthReduction
from defence.feature_distillation_jpeg import FD_jpeg
from third_party.TransferAttack.transferattack.attack_map import attack_zoo

class AccuracyMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_clean = 0
        self.correct_adv = 0
        self.total_correct_clean = 0
        self.total_correct_adv = 0
        self.total_num = 0
        self.clean_acc = 0
        self.adv_acc = 0
        self.attack_rate = 0

    def update(self, correct_clean, correct_adv, total_num=1):
        self.correct_clean = correct_clean
        self.correct_adv = correct_adv
        self.total_correct_clean += correct_clean
        self.total_correct_adv += correct_adv
        self.total_num += total_num
        self.clean_acc = self.total_correct_clean / self.total_num
        self.adv_acc = self.total_correct_adv / self.total_num
        if self.clean_acc != 0:
            self.attack_rate = 1 - self.adv_acc / self.clean_acc
        else:
            self.attack_rate = 1 - self.adv_acc / (self.clean_acc + 1e-12)


def tranferability_validate_v2(args,
                     loader,
                     target_model_list,
                     suggrogate_model,
                     device,
                     trans,
                     epoch,
                     writer=None):
    """
    Runs an ensemble adversarial attack and eval the generated adversarial example's attack tranferability .

    Args:
        args (object) : an arguments object (see
            :meth:`~robustness.train.train_model` for list of arguments
        
        loader (iterable) : an iterable loader of the form 
            `(image_batch, label_batch)`
        model (AttackerModel) : target model
        opt (ch.optim.Optimizer) : optimizer to use (ignored for evaluation)
        epoch (int) : which epoch we are currently on
        adv (bool) : whether to evaluate adversarially (otherwise standard)
        writer : tensorboardX writer (optional)
        loop_type ('train' or 'val') : whether we are training or evaluating

    Returns:
        The average top1 accuracy and the average loss across the epoch.
    """
    suggrogate_model.zero_grad()
    suggrogate_model.eval()
    metrix = []
    for target_model in target_model_list:
        target_model.eval()
        metrix.append(AccuracyMeter())

    prec = 'AdvPrec'
    loop_msg = 'Val'

    train_criterion = torch.nn.CrossEntropyLoss()
    if args.defence_type == 'jpeg':
        defender = Jpeg_compression()
    elif args.defence_type == 'bit_depth':
        defender = BitDepthReduction()
    elif args.defence_type == 'fd_jpeg':
        defender = FD_jpeg()
    attacker = attack_zoo[args.attack_type](suggrogate_model,random_restart = True,epoch = args.trans_attack_steps,epsilon = args.trans_eps,alpha = args.trans_attack_lr)
    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        perb = attacker(inp,target)
        adv_img = inp.detach()+perb

        if args.defence_type == 'jpeg' or args.defence_type == 'bit_depth' or args.defence_type == 'fd_jpeg':
            adv_img = defender(adv_img).cuda()        
        # input_tensor = inp[0].clone().detach().to(torch.device('cpu'))
        # input_tensor = torch.unsqueeze(input_tensor, dim=0)
        # save_image(input_tensor,'/private/workpace/ensrobusttraining/visualization/eps16/saved_clean_image{num}.png'.format(num = i))
        # input_tensor = adv_img[0].clone().detach().to(torch.device('cpu'))
        # input_tensor = torch.unsqueeze(input_tensor, dim=0)
        # save_image(input_tensor,'/private/workpace/ensrobusttraining/visualization/eps16/saved_adv_image{num}.png'.format(num = i))

        for idx, model in enumerate(target_model_list):
            with torch.no_grad():
                r_clean = model(inp)
                r_adv = model(adv_img)
            # clean
            pred_clean = r_clean.max(1)[1]
            correct_clean = (pred_clean == target).sum().item()
            # adv
            pred_adv = r_adv.max(1)[1]
            correct_adv = (pred_adv == target).sum().item()
            metrix[idx].update(correct_clean, correct_adv, target.size(0))
        progress_bar = ('TRANS VAL | Epoch: {epoch} | ASR: {asr:.3f} | NatAcc: {natacc:.3f} | AdvAcc {advacc:.3f}'.format(epoch = epoch, 
                                                                                                                          asr = metrix[0].attack_rate, 
                                                                                                                          natacc = metrix[0].clean_acc, 
                                                                                                                          advacc = metrix[0].adv_acc))
        iterator.set_description(progress_bar)
        iterator.refresh()
    results = PrettyTable(field_names=["Model name", "Nat. Acc. (%)", "Adv. Acc. (%)", "ASR. (%)"])   
    for idx in range(len(target_model_list)):
        writer.add_scalar('Transferability eval/Nat. Acc. (%)', metrix[idx].clean_acc, epoch)
        writer.add_scalar('Transferability eval/Adv. Acc. (%)', metrix[idx].adv_acc, epoch)
        writer.add_scalar('Transferability eval/ASR. (%)', metrix[idx].attack_rate, epoch)
        results.add_row([
            f"model{idx}",
            f"{str(round(metrix[idx].clean_acc * 100, 2)).ljust(13, ' ')}",
            f"{str(round(metrix[idx].adv_acc * 100, 2)).ljust(13, ' ')}",
            f"{str(round(metrix[idx].attack_rate * 100, 2)).ljust(8, ' ')}",
        ])
    print(results)

    return metrix
