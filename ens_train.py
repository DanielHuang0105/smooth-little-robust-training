# 直接一个弱鲁棒AT+cosine similarity regularization 可不可以 -> 好像不行，因为它已经smooth了
# 直接训练 resnet_l2_0.05(train) + vgg11 + inceptionv3 黑盒攻击模型densenet121 
#   loss来源，三种方法： model(inputs_clean)+model(inputs_adv) model(inputs_adv) model(inputs_self_adv)+model(inputs_adv)
#   loss集成，两种方法： fuse logits fuse losses
# 直接进行cosine正则
# 将输入模型变为 ‘微鲁棒模型’+LGV模型
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import sys
import copy
from tqdm import tqdm
from scipy.stats import truncnorm

def Cosine(g1, g2):
	return torch.abs(F.cosine_similarity(g1, g2)).mean()  # + (0.05 * torch.sum(g1**2+g2**2,1)).mean()

def grad_cosine(loss_adv,loss_clean, inputs_adv,inputs_clean):
    bs = inputs_clean.size(0)

    grad_clean = torch.autograd.grad(loss_clean, inputs_clean, retain_graph=True)[0]
    grad_adv = torch.autograd.grad(loss_adv, inputs_adv, retain_graph=True)[0]
    g_clean = grad_clean.view(bs, -1)
    g_adv = grad_adv.view(bs, -1)
    cossim = Cosine(g_clean,g_adv).sum()/bs

    return cossim.item()

def PGD(args,model, clean_img, clean_target):
    images = clean_img.clone().detach().cuda()
    labels = clean_target.clone().detach().cuda()

    if args.targeted:
        pass

    loss = nn.CrossEntropyLoss()

    adv_images = images.clone().detach()
    batch_size = len(images)

    if args.random_start:
        # Starting at a uniformly random point
        delta = torch.empty_like(adv_images).normal_()
        d_flat = delta.view(adv_images.size(0), -1) 
        n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*args.eps  
        adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

    for _ in range(args.attack_steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        # Calculate loss
        if args.targeted:
            # cost = -loss(outputs, target_labels)
            pass
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]
        grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + args.eps_for_division  
        grad = grad / grad_norms.view(batch_size, 1, 1, 1)  
        adv_images = adv_images.detach() + args.attack_lr * grad
        
        delta = adv_images - images
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)  
        factor = args.eps / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def adv_train_withclean_withasam(trainloader, criterion, minimizer, model, zoo_models, writer, epoch, args):

    losses_combine = AverageMeter()
    top1_combine = AverageMeter()
    losses_clean = AverageMeter()
    top1_clean = AverageMeter()
    losses_adv = AverageMeter()
    top1_adv = AverageMeter()
   
    # training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    iterator = tqdm(enumerate(trainloader), total=len(trainloader))
    
    for i, (inputs_clean, targets_clean) in iterator:

        # inputs
        inputs_clean, targets_clean = inputs_clean.to(device), targets_clean.to(device)

        adv_generating_model = copy.deepcopy(model)
        # the model generate adv should be in eval() model
        adv_generating_model.eval()

        # generate adv images         
        inputs_adv = PGD(args, adv_generating_model, inputs_clean, targets_clean)

        # training 
        ## in case that the adv_generating_model is the training model itself, clean the gradient and swith the model
        model.zero_grad()
        model.train()
        # clean image
        logits_clean = model(inputs_clean)
        loss1 = criterion(logits_clean, targets_clean).mean()

        logits_adv = model(inputs_adv)
        loss2 = criterion(logits_adv, targets_clean).mean()
        
        if args.loss_schema == 'averaged':
            loss = args.fuse_weight*loss1+(1-args.fuse_weight)*loss2
        elif args.loss_schema == 'weighted':
            loss = (1 / (1+args.fuse_weight)) * (loss2 + args.fuse_weight* loss1)

        # ascend step
        loss.backward()
        minimizer.ascent_step()

        # descent step
        if args.loss_schema == 'averaged':
            ((1-args.fuse_weight)*criterion(model(inputs_adv),targets_clean).mean()+args.fuse_weight*criterion(model(inputs_clean),targets_clean).mean()).backward()
        elif args.loss_schema == 'weighted':
            ((1 / (1+args.fuse_weight))*(criterion(model(inputs_adv),targets_clean).mean()+args.fuse_weight*criterion(model(inputs_clean),targets_clean).mean())).backward()
        minimizer.descent_step()
        
        # clean
        acc1, _ = accuracy(logits_clean, targets_clean, topk=(1, 5))
        losses_clean.update(loss1.item(), inputs_clean.size(0))
        top1_clean.update(acc1[0], inputs_clean.size(0))        
        # adv
        acc2, _ = accuracy(logits_adv, targets_clean, topk=(1, 5))
        losses_adv.update(loss2.item(), inputs_clean.size(0))
        top1_adv.update(acc2[0], inputs_clean.size(0))


        # return losses_clean, top1_clean, losses_adv, top1_adv, losses_combine, top1_combine

        progress_bar = ('Train | Epoch: {epoch} | loss_clean: {loss_clean:.3f} | top1_clean: {top1_clean:.3f} | loss_adv: {loss_adv:.3f} | top1_adv: {top1_adv:.3f}'.format(epoch = epoch, 
                                                                                                                                                                            loss_clean = losses_clean.avg, 
                                                                                                                                                                            top1_clean = top1_clean.avg, 
                                                                                                                                                                            loss_adv = losses_adv.avg, 
                                                                                                                                                                            top1_adv = top1_adv.avg))
        iterator.set_description(progress_bar)
        iterator.refresh()
    writer.add_scalar('Train/Loss_adv', losses_adv.val, epoch)
    writer.add_scalar('Train/Prec@1_adv', top1_adv.val, epoch)
    writer.add_scalar('Train/Loss_clean', losses_clean.val, epoch)
    writer.add_scalar('Train/Prec@1_clean', top1_clean.val, epoch)
def normal_train(trainloader, criterion, optimizer, model, zoo_models, writer, epoch, args):

    losses_clean = AverageMeter()
    top1_clean = AverageMeter()
   
    # training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    iterator = tqdm(enumerate(trainloader), total=len(trainloader))
    
    for i, (inputs_clean, targets_clean) in iterator:

        # inputs
        inputs_clean, targets_clean = inputs_clean.to(device), targets_clean.to(device)
        model.zero_grad()
        model.train()

        # descend step
        logits_clean = model(inputs_clean)
        loss2 = criterion(logits_clean, targets_clean)
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()

        # adv
        acc2, _ = accuracy(logits_clean, targets_clean, topk=(1, 5))
        losses_clean.update(loss2.item(), inputs_clean.size(0))
        top1_clean.update(acc2[0], inputs_clean.size(0))


        # return losses_clean, top1_clean, losses_adv, top1_adv, losses_combine, top1_combine

        progress_bar = ('Train | Epoch: {epoch} | loss: {losses:.3f} | Top1: {top1:.3f}'.format(epoch = epoch, losses = losses_clean.avg, top1 = top1_clean.avg))
        iterator.set_description(progress_bar)
        iterator.refresh()

    writer.add_scalar('Train/Loss_adv', losses_clean.val, epoch)
    writer.add_scalar('Train/Prec@1_adv', top1_clean.val, epoch)