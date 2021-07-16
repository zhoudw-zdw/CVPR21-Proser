import argparse
import os
import os.path as osp
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import MultiStepLR
from models import *
from utils import ensure_path, progress_bar
from models.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, compute_confidence_interval, one_hot,Identity
from torch.distributions import Categorical
import random

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if args.shmode==False:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(closerloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if args.shmode==False:
                progress_bar(batch_idx, len(closerloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, osp.join(args.save_path,'ckpt.pth'))
        best_acc = acc


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model_type', default='softmax', type=str, help='Recognition Method')
    parser.add_argument('--backbone', default='WideResnet', type=str, help='Backbone type.')
    parser.add_argument('--dataset', default='cifar10_relabel',type=str,help='dataset configuration')
    parser.add_argument('--gpu', default='2',type=str,help='use gpu')
    parser.add_argument('--known_class', default=6,type=int,help='number of known class')
    parser.add_argument('--seed', default='9',type=int,help='random seed for dataset generation.')
    parser.add_argument('--shmode',action='store_true')
    
    args = parser.parse_args()
    pprint(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] =args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing data..')
    if args.dataset=='cifar10_relabel':
        from data.cifar10_relabel import CIFAR10 as Dataset
    
    trainset=Dataset('train',seed=args.seed)
    knownlist,unknownlist=trainset.known_class_show()
    trainloader=torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    closeset=Dataset('testclose',seed=args.seed)
    closerloader=torch.utils.data.DataLoader(closeset, batch_size=512, shuffle=True, num_workers=4)
    openset=Dataset('testopen',seed=args.seed)
    openloader=torch.utils.data.DataLoader(openset, batch_size=512, shuffle=True, num_workers=4)
    
    print('==> Building model..')
    if args.backbone=='WideResnet':
        net=Wide_ResNet(28, 10, 0.3, args.known_class)
    net = net.to(device)
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)

    save_path1 = osp.join('results','D{}-M{}-B{}'.format(args.dataset,args.model_type, args.backbone,))
    save_path2 = 'LR{}-K{}-U{}-Seed{}'.format(str(args.lr), knownlist,unknownlist,str(args.seed))
    args.save_path = osp.join(save_path1, save_path2)
    ensure_path(save_path1, remove=False)
    ensure_path(args.save_path, remove=False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    scheduler = MultiStepLR(optimizer, milestones=[50,125], gamma=0.1)
    for epoch in range(start_epoch, start_epoch+200):
        scheduler.step()
        train(epoch)
        test(epoch)
        if (epoch+1)%10==0:
            state = {'net': net.state_dict(),'epoch': epoch,}
            torch.save(state, osp.join(args.save_path,'Modelof_Epoch'+str(epoch)+'.pth'))
