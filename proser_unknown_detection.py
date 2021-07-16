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
import copy
import random

def traindummy(epoch,net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr*0.01,momentum=0.9, weight_decay=5e-4)

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha=args.alpha
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        totallenth=len(inputs)
        halflenth=int(len(inputs)/2)
        beta=torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        
        prehalfinputs=inputs[:halflenth]
        prehalflabels=targets[:halflenth]
        laterhalfinputs=inputs[halflenth:]
        laterhalflabels=targets[halflenth:]

        index = torch.randperm(prehalfinputs.size(0)).cuda()
        pre2embeddings=pre2block(net,prehalfinputs)
        mixed_embeddings = beta * pre2embeddings + (1 - beta) * pre2embeddings[index]

        dummylogit=dummypredict(net,laterhalfinputs)
        lateroutputs=net(laterhalfinputs)
        latterhalfoutput=torch.cat((lateroutputs,dummylogit),1)
        prehalfoutput=torch.cat((latter2blockclf1(net,mixed_embeddings),latter2blockclf2(net,mixed_embeddings)),1)
        
        maxdummy,_=torch.max(dummylogit.clone(),dim=1)
        maxdummy=maxdummy.view(-1,1)
        dummpyoutputs=torch.cat((lateroutputs.clone(),maxdummy),dim=1)
        for i in range(len(dummpyoutputs)):
            nowlabel=laterhalflabels[i]
            dummpyoutputs[i][nowlabel]=-1e9
        dummytargets=torch.ones_like(laterhalflabels)*args.known_class
        

        outputs=torch.cat((prehalfoutput,latterhalfoutput),0)
        loss1= criterion(prehalfoutput, (torch.ones_like(prehalflabels)*args.known_class).long().cuda()) 
        loss2=criterion(latterhalfoutput,laterhalflabels )
        loss3= criterion(dummpyoutputs, dummytargets)
        loss=0.01*loss1+args.lamda1*loss2+args.lamda2*loss3
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.shmode==False:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) L1 %.3f, L2 %.3f'\
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total , loss1.item(), loss2.item(), ))
        

def valdummy(epoch,net,mainepoch):
    net.eval()
    CONF_AUC=False
    CONF_DeltaP=True
    auclist1=[]
    auclist2=[]
    linspace=[0]
    closelogits=torch.zeros((len(closeset),args.known_class+1)).cuda()
    openlogits=torch.zeros((len(openset),args.known_class+1)).cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(closerloader):
            inputs, targets = inputs.to(device), targets.to(device)
            batchnum=len(targets)
            logits=net(inputs)
            dummylogit=dummypredict(net,inputs)
            maxdummylogit,_=torch.max(dummylogit,1)
            maxdummylogit=maxdummylogit.view(-1,1)
            totallogits=torch.cat((logits,maxdummylogit),dim=1)
            closelogits[batch_idx*batchnum:batch_idx*batchnum+batchnum,:]=totallogits
        for batch_idx, (inputs, targets) in enumerate(openloader):
            inputs, targets = inputs.to(device), targets.to(device)
            batchnum=len(targets)
            logits=net(inputs)
            dummylogit=dummypredict(net,inputs)
            maxdummylogit,_=torch.max(dummylogit,1)
            maxdummylogit=maxdummylogit.view(-1,1)
            totallogits=torch.cat((logits,maxdummylogit),dim=1)
            openlogits[batch_idx*batchnum:batch_idx*batchnum+batchnum,:]=totallogits
    Logitsbatchsize=200
    maxauc=0
    maxaucbias=0
    for biasitem in linspace:
        if CONF_AUC:
            for temperature in [1024.0]:
                closeconf=[]
                openconf=[]
                closeiter=int(len(closelogits)/Logitsbatchsize)
                openiter=int(len(openlogits)/Logitsbatchsize)
                for batch_idx  in range(closeiter):
                    logitbatch=closelogits[batch_idx*Logitsbatchsize:batch_idx*Logitsbatchsize+Logitsbatchsize,:]
                    logitbatch[:,-1]=logitbatch[:,-1]+biasitem
                    embeddings=nn.functional.softmax(logitbatch/temperature,dim=1)
                    conf=embeddings[:,-1]
                    closeconf.append(conf.cpu().numpy())
                closeconf=np.reshape(np.array(closeconf),(-1))
                closelabel=np.ones_like(closeconf)
                for batch_idx  in range(openiter):
                    logitbatch=openlogits[batch_idx*Logitsbatchsize:batch_idx*Logitsbatchsize+Logitsbatchsize,:]
                    logitbatch[:,-1]=logitbatch[:,-1]+biasitem
                    embeddings=nn.functional.softmax(logitbatch/temperature,dim=1)
                    conf=embeddings[:,-1]
                    openconf.append(conf.cpu().numpy())
                openconf=np.reshape(np.array(openconf),(-1))
                openlabel=np.zeros_like(openconf)
                totalbinary=np.hstack([closelabel,openlabel])
                totalconf=np.hstack([closeconf,openconf])
                auc1=roc_auc_score(1-totalbinary,totalconf)
                auc2=roc_auc_score(totalbinary,totalconf)
                print('Temperature:',temperature,'bias',biasitem,'AUC_by_confidence',auc2)
                auclist1.append(np.max([auc1,auc2]))
        if CONF_DeltaP:
            for temperature in [1024.0]:
                closeconf=[]
                openconf=[]
                closeiter=int(len(closelogits)/Logitsbatchsize)
                openiter=int(len(openlogits)/Logitsbatchsize)
                for batch_idx  in range(closeiter):
                    logitbatch=closelogits[batch_idx*Logitsbatchsize:batch_idx*Logitsbatchsize+Logitsbatchsize,:]
                    logitbatch[:,-1]=logitbatch[:,-1]+biasitem
                    embeddings=nn.functional.softmax(logitbatch/temperature,dim=1)
                    dummyconf=embeddings[:,-1].view(-1,1)
                    maxknownconf,_=torch.max(embeddings[:,:-1],dim=1)
                    maxknownconf=maxknownconf.view(-1,1)
                    conf=dummyconf-maxknownconf
                    closeconf.append(conf.cpu().numpy())
                closeconf=np.reshape(np.array(closeconf),(-1))
                closelabel=np.ones_like(closeconf)
                for batch_idx  in range(openiter):
                    logitbatch=openlogits[batch_idx*Logitsbatchsize:batch_idx*Logitsbatchsize+Logitsbatchsize,:]
                    logitbatch[:,-1]=logitbatch[:,-1]+biasitem
                    embeddings=nn.functional.softmax(logitbatch/temperature,dim=1)
                    dummyconf=embeddings[:,-1].view(-1,1)
                    maxknownconf,_=torch.max(embeddings[:,:-1],dim=1)
                    maxknownconf=maxknownconf.view(-1,1)
                    conf=dummyconf-maxknownconf
                    openconf.append(conf.cpu().numpy())
                openconf=np.reshape(np.array(openconf),(-1))
                openlabel=np.zeros_like(openconf)
                totalbinary=np.hstack([closelabel,openlabel])
                totalconf=np.hstack([closeconf,openconf])
                auc1=roc_auc_score(1-totalbinary,totalconf)
                auc2=roc_auc_score(totalbinary,totalconf)
                print('Temperature:',temperature,'bias',biasitem,'AUC_by_Delta_confidence',auc1)
                auclist1.append(np.max([auc1,auc2]))
    return np.max(np.array(auclist1))

def finetune_proser(epoch=59): 
    print('Now processing epoch',epoch)
    net=getmodel(args)
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(model_path), 'Error: no checkpoint directory found!'
    modelname='Modelof_Epoch'+str(epoch)+'.pth'
    checkpoint = torch.load(osp.join(model_path,save_path2,modelname))
    net.load_state_dict(checkpoint['net'])

    net.clf2=nn.Linear(640,args.dummynumber)
    net=net.cuda()

    FineTune_MAX_EPOCH=10
    wholebestacc=0
    for finetune_epoch in range(FineTune_MAX_EPOCH):
        traindummy(finetune_epoch,net)
        if (finetune_epoch+1)%10==0:
            finetuneacc=valdummy(finetune_epoch,net,epoch)
    return wholebestacc

def dummypredict(net,x):
    if  args.backbone=="WideResnet":
        out = net.conv1(x)
        out = net.layer1(out)
        out = net.layer2(out)
        out = net.layer3(out)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.clf2(out)
        return out

def pre2block(net,x):
    if  args.backbone=="WideResnet":
        out = net.conv1(x)
        out = net.layer1(out)
        out = net.layer2(out)
        return out

def latter2blockclf1(net,x):
    if  args.backbone=="WideResnet":
        out = net.layer3(x)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.linear(out)
        return out

def latter2blockclf2(net,x):
    if args.backbone=="WideResnet":
        out = net.layer3(x)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.clf2(out)
        return out

def getmodel(args):
    print('==> Building model..')
    if args.backbone=='WideResnet':
        net=Wide_ResNet(28, 10, 0.3, args.known_class)
    net=net.cuda()
    return net


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model_type', default='Proser', type=str, help='Recognition Method')
    parser.add_argument('--backbone', default='WideResnet', type=str, help='Backbone type.')
    parser.add_argument('--dataset', default='cifar10_relabel',type=str,help='dataset configuration')
    parser.add_argument('--gpu', default='0',type=str,help='use gpu')
    parser.add_argument('--known_class', default=6,type=int,help='number of known class')
    parser.add_argument('--seed', default='9',type=int,help='random seed for dataset generation.')
    parser.add_argument('--lamda1', default='1',type=float,help='trade-off between loss')
    parser.add_argument('--lamda2', default='1',type=float,help='trade-off between loss')
    parser.add_argument('--alpha', default='1',type=float,help='alpha value for beta distribution')
    parser.add_argument('--dummynumber', default=1,type=int,help='number of dummy label.')
    parser.add_argument('--shmode',action='store_true')
    
    args = parser.parse_args()
    pprint(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] =args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  
    start_epoch = 0  
    
    print('==> Preparing data..')
    if args.dataset=='cifar10_relabel':
        from data.cifar10_relabel import CIFAR10 as Dataset

    trainset=Dataset('train',seed=args.seed)
    knownlist,unknownlist=trainset.known_class_show()
    trainloader=torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    closeset=Dataset('testclose',seed=args.seed)
    closerloader=torch.utils.data.DataLoader(closeset, batch_size=500, shuffle=True, num_workers=4)
    openset=Dataset('testopen',seed=args.seed)
    openloader=torch.utils.data.DataLoader(openset, batch_size=500, shuffle=True, num_workers=4)

    
    save_path1 = osp.join('results','D{}-M{}-B{}'.format(args.dataset,args.model_type, args.backbone,))
    model_path = osp.join('results','D{}-M{}-B{}'.format(args.dataset,'softmax', args.backbone,))
    save_path2 = 'LR{}-K{}-U{}-Seed{}'.format(str(args.lr), knownlist,unknownlist,str(args.seed))
    args.save_path = osp.join(save_path1, save_path2)
    ensure_path(save_path1, remove=False)
    ensure_path(args.save_path, remove=False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    globalacc=0
    finetune_proser(59)
    