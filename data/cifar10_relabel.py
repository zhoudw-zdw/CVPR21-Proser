import os.path as osp
import PIL
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as sio
import torch
import copy

class CIFAR10(Dataset):
    def __init__(self, setname,seed ):
        self.known_class=6
        self.total_class=10
        self.setname=setname
        #random choose knwon and unknown class
        np.random.seed(seed)
        self.total_classes_perm=np.arange(self.total_class)
        np.random.shuffle(self.total_classes_perm)
        
        self.known_class_list=self.total_classes_perm[:self.known_class]
        self.unknon_class_list=self.total_classes_perm[self.known_class:]
        print('Known class list:',self.known_class_list,'Unknown class list',self.unknon_class_list)
        
        #load dataset, pick corresponding classes and form dataset.
        self.traindata=sio.loadmat('./data/cifar/data_batch_1.mat')
        self.trainx1=self.traindata['data']
        self.trainy1=self.traindata['labels']
        self.traindata=sio.loadmat('./data/cifar/data_batch_2.mat')
        self.trainx2=self.traindata['data']
        self.trainy2=self.traindata['labels']
        self.traindata=sio.loadmat('./data/cifar/data_batch_3.mat')
        self.trainx3=self.traindata['data']
        self.trainy3=self.traindata['labels']
        self.traindata=sio.loadmat('./data/cifar/data_batch_4.mat')
        self.trainx4=self.traindata['data']
        self.trainy4=self.traindata['labels']
        self.traindata=sio.loadmat('./data/cifar/data_batch_5.mat')
        self.trainx5=self.traindata['data']
        self.trainy5=self.traindata['labels']

        self.trainx=np.vstack([self.trainx1,self.trainx2,self.trainx3,self.trainx4,self.trainx5])
        self.trainy=np.vstack([self.trainy1,self.trainy2,self.trainy3,self.trainy4,self.trainy5])
        assert(len(self.trainx)==len(self.trainy))
        
        self.testdata=sio.loadmat('./data/cifar/test_batch.mat')
        self.testx=self.testdata['data']
        self.testy=self.testdata['labels']

        #relabel dataset
        self.knowndict={}
        self.unknowndict={}
        for i in range(len(self.known_class_list)):
            self.knowndict[self.known_class_list[i]]=i
        for j in range(len(self.unknon_class_list)):
            self.unknowndict[self.unknon_class_list[j]]=j+len(self.known_class_list)
        if setname=='train':
            print(self.knowndict,self.unknowndict)

        self.copytrainy=copy.deepcopy(self.trainy)
        self.copytesty=copy.deepcopy(self.testy)
        for i in range(len(self.known_class_list)):
            self.trainy[self.copytrainy==self.known_class_list[i]]=self.knowndict[self.known_class_list[i]]
            self.testy[self.copytesty==self.known_class_list[i]]=self.knowndict[self.known_class_list[i]]
        for j in range(len(self.unknon_class_list)):
            self.trainy[self.copytrainy==self.unknon_class_list[j]]=self.unknowndict[self.unknon_class_list[j]]
            self.testy[self.copytesty==self.unknon_class_list[j]]=self.unknowndict[self.unknon_class_list[j]]
        self.origin_known_list=self.known_class_list
        self.origin_unknown_list=self.unknon_class_list
        self.new_known_list=np.arange(self.known_class)
        self.new_unknown_list=np.arange(self.known_class,self.known_class+len(self.unknon_class_list))


        self.trian_data_known_index=[]
        self.test_data_known_index=[]
        for item in self.new_known_list:
            index=np.where(self.trainy==item)
            index=list(index[0])
            self.trian_data_known_index=self.trian_data_known_index+index
            index=np.where(self.testy==item)
            index=list(index[0])
            self.test_data_known_index=self.test_data_known_index+index
        
        self.train_data_index_perm=np.arange(len(self.trainy))
        self.train_data_unknown_index=np.setdiff1d(self.train_data_index_perm,self.trian_data_known_index)
        self.test_data_index_perm=np.arange(len(self.testy))
        self.test_data_unknown_index=np.setdiff1d(self.test_data_index_perm,self.test_data_known_index)
        
        assert (len(self.test_data_unknown_index)+len(self.test_data_known_index)==len(self.testy))


        self.transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.trainx=torch.tensor(self.trainx).view(-1,3,32,32) /255.0
        self.testx=torch.tensor(self.testx).view(-1,3,32,32) /255.0
        self.trainy=torch.tensor(self.trainy).view(-1).long()
        self.testy=torch.tensor(self.testy).view(-1).long()

        if setname=="train":
            self.datax=(self.trainx[self.trian_data_known_index]).float()
            self.datay=(self.trainy[self.trian_data_known_index]).long()
        elif setname=="testclose":
            self.datax=(self.testx[self.test_data_known_index]).float()
            self.datay=(self.testy[self.test_data_known_index]).long()
        elif setname=="testopen":
            self.datax=(self.testx[self.test_data_unknown_index]).float()
            self.datay=(self.testy[self.test_data_unknown_index]).long()
        
        
    def __len__(self):
        return len(self.datay)

    def known_class_show(self):
        return self.origin_known_list,self.origin_unknown_list
    def __getitem__(self, index):
        if self.setname=='train':
            data, label = self.transform_train(self.datax[index]), self.datay[index]
        else:
            data, label = self.transform_test(self.datax[index]), self.datay[index]
        return data, label 
        
