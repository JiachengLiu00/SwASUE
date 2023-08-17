import argparse
import os
#model

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate,get_test_transformations

from utils.collate import collate_custom
from termcolor import colored

from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from sampler import SubsetSequentialSampler
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from models.resnet_cifar import resnet18
from models.models import ActiveModel
import random


def AcquisitionFunction(p,data,subset,device,num_class,n_query):
    num = int(n_query/num_class)
    model = get_model(p, p['scan_model'])
    model = model.to(device)

    pretrain_path = p['selflabel_model']
    subset_loader = DataLoader(data, batch_size=128,
                                    sampler=SubsetSequentialSampler(subset),
                                    pin_memory=False, drop_last=True)

    output = torch.tensor([]).to(device)
    softmax = nn.Softmax(dim=1)

    for i in subset_loader:
        with torch.no_grad():
            input = i['image']
            input = input.to(device)
            output_batch = model(input)[0]
            output_batch = softmax(output_batch)
            output = torch.cat((output, output_batch), 0)

    pro,index =torch.max(output, dim = 1)
    pro = pro.detach().cpu().numpy()
    index = index.detach().cpu().numpy()
    cluster = [[] for i in range(num_class)]
    for i in range(num_class):
        cluster[i] = list(np.where(index == i))
    class_pro  = [[] for i in range(num_class)]
    for j in range(len(cluster)):
        class_pro[j]=np.squeeze(pro[cluster[j]],axis=0)
    xiabiao = [[] for i in range(num_class)]


    query_idxs = np.array([], dtype=int)
    for k in range (num_class):
        xiabiao[k] = -np.array(class_pro[k])
        xiabiao[k] = np.argsort(xiabiao[k])[0:num]
        query_idxs = np.concatenate((query_idxs, cluster[k][0][xiabiao[k]]))

    query_set = list(np.array(subset)[query_idxs])
    remainSet = set(np.array(subset)) - set(np.array(subset)[query_idxs])
    remainSet = list(remainSet)
    return query_set,remainSet

def uncertainty_sample(device,model,subset,dataset,n_query :int = 1000):
    model.eval()
    subset_loader = DataLoader(dataset, batch_size=128,
                                    sampler=SubsetSequentialSampler(subset),
                                    pin_memory=True, drop_last=True)
    output_w= torch.tensor([]).to(device)
    output_s = torch.tensor([]).to(device)
    softmax = nn.Softmax(dim=1)
    for i in subset_loader:
        with torch.no_grad():
            input = i['image']
            input = input.to(device)

            input_augmented = i['image_augmented']
            input_augmented = input_augmented.to(device)

            output_batch,_ = model(input)
            output_batch = softmax(output_batch)
            output_w = torch.cat((output_w, output_batch), 0)

            output_augmented_batch,_ = model(input_augmented)
            output_augmented_batch = softmax(output_augmented_batch)
            output_s = torch.cat((output_s, output_augmented_batch), 0)

    output_w = output_w.cpu().numpy()
    output_s = output_s.cpu().numpy()

    pc_w = output_w
    pc_s = output_s
    H_w_s = (-pc_w * np.log(pc_s + 1e-10)).sum(
        axis=-1
    )
    H_s_w = (-pc_s * np.log(pc_w + 1e-10)).sum(
        axis=-1
    )
    H = H_w_s + H_s_w
    idx = (-H).argsort()[:n_query]

    query_set = list(np.array(subset)[idx])
    remainSet = set(np.array(subset)) - set(np.array(subset)[idx])
    remainSet = list(remainSet)

    return query_set,remainSet

def active_train(model,epoch,train_dataset,loss_function,optimizer,scheduler,device):
    model.train()

    for k in range(epoch):

        for i in tqdm(train_dataset,leave=False):
            images = i['image']
            images_augmented = i['image_augmented']
            target = i['target']

            images = images.to(device)
            images_augmented = images_augmented.to(device)
            target = target.to(device)

            output_w,_ = model(images)
            output_s,_ = model(images_augmented)

            loss_y = torch.mean((output_w-output_s)**2)
            loss_w = loss_function(output_w, target)
            loss_s = loss_function(output_s, target)
            loss = loss_w+ loss_s + loss_y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
    print(colored('-->Finished.','red'))

def t_accuracy(model,data,device):
    model.eval()
    acc_sum, n = 0.0, 0
    for i in data:
        image = i['image'].to(device)
        target = i['target'].to(device)
        with torch.no_grad():

            # x =x.unsqueeze(0)
            pred,_ = model(image)
            pred = pred.argmax(dim=1)
            acc_sum  +=  pred.eq(target).sum().item()
        n += target.shape[0]
    return acc_sum / n * 100


FLAGS = argparse.ArgumentParser(description='SCAN Loss')
FLAGS.add_argument('--config_env', help='Location of path config file',default='./configs/env.yml')
FLAGS.add_argument('--config_exp', help='Location of experiments config file',default='./configs/selflabel/selflabel_svhn.yml')
FLAGS.add_argument('--seed', type=int, default=1, help='Random seed')
FLAGS.add_argument('--num_clusters', type=int, default=10, help='Number of clusters')
FLAGS.add_argument('--pre_train',type=bool,default=False,help='pre-training')
FLAGS.add_argument('--cycle',type=int,default=5,help='Number of iterations')
FLAGS.add_argument('--epoch',type=int,default=100,help='Rounds of training')
FLAGS.add_argument('--budget',type=int,default=1000,help='Rounds of budget')
FLAGS.add_argument('--lr',type=float,default=0.01,help='Learning rate')
FLAGS.add_argument('--weight_decay',type=float,default=0.0001,help='Weight decay')
FLAGS.add_argument('--momentum',type=float,default=0.9,help='Weight decay')
FLAGS.add_argument('--milestones',default=[120,180],help='Weight decay')



def main():
    args = FLAGS.parse_args()
    p = create_config(args.config_env, args.config_exp, args.seed, args.num_clusters)
    print(colored(p, 'red'))
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # model
    backbone = resnet18()
    model = ActiveModel(backbone,p['num_classes'])
    model = model.to(device)
    if args.pre_train:
        state = torch.load(p['selflabel_model'], map_location='cpu')
        model.load_state_dict(state,strict=False)
        model = torch.nn.DataParallel(model)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))

    # Transforms
    strong_transforms =get_train_transformations(p)
    val_transforms = get_val_transformations(p)
    test_transforms = get_test_transformations(p)

    print(colored(f'strong_transforms:{strong_transforms}','blue'))

    #data
    train_dataset_ = get_train_dataset(p, {'standard': val_transforms, 'augment': strong_transforms},
                                      split='train', to_augmented_dataset=True)
    test_dataset_ = get_val_dataset(p, test_transforms)
    test_dataset = torch.utils.data.DataLoader(test_dataset_ , num_workers=8,
                                batch_size=128, pin_memory=False, collate_fn=collate_custom,
                                drop_last=False, shuffle=False)


    ADDNUM = 10000
    n_dataset = len(train_dataset_)
    label_set = []
    unlabel_set = [x for x in range(n_dataset)]
    random.shuffle(unlabel_set)
    subset = unlabel_set[:ADDNUM]
    all_set = unlabel_set

    #initial data
    if args.pre_train:
        print(colored('acquire initial data','blue'))
        query_set,remainSet = AcquisitionFunction(p, train_dataset_,subset,device,p['num_classes'],args.budget)
        label_set += query_set
        unlabel_set = unlabel_set[ADDNUM:] + remainSet
    else:
        query_set = unlabel_set[:1000]
        label_set += query_set
        unlabel_set = unlabel_set[1000:]

    label_set_loader = DataLoader(train_dataset_, batch_size=128,
                                   sampler=SubsetRandomSampler(label_set),
                                   pin_memory=True, drop_last=True)
    print(colored(f'the number of initial data:{len(label_set)}','cyan'))

    #loss function
    loss_function = nn.CrossEntropyLoss(reduction='mean')

    #optimizer
    if args.pre_train:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #lr=0.0001, weight_decay=0.0001
        sched_backbone = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)#cifar10[5,20],cifar100,svhn[25,50]
    else:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay) #params=model.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4
        sched_backbone = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)  # [120,180]

    print(colored('Start Training', 'yellow'))
    pre_acc = []

    for i in range(args.cycle):

        active_train(model,args.epoch,label_set_loader,loss_function,optimizer,sched_backbone,device)
        acc = t_accuracy(model,test_dataset,device)
        print(f'Accuracy of Cycle {i+1}:',colored(acc, 'red'))
        if i < (args.epoch) :
            random.shuffle(unlabel_set)
            subset = unlabel_set[:ADDNUM]
            query_set, remainSet = uncertainty_sample(device, model, subset, train_dataset_, n_query=args.budget)
            label_set += query_set
            unlabel_set = unlabel_set[ADDNUM:] + remainSet
            label_set_loader = DataLoader(train_dataset_, batch_size=128,
                                          sampler=SubsetRandomSampler(label_set),
                                          pin_memory=True, drop_last=True)
        pre_acc.append(acc)
    pre_acc = np.array(pre_acc)
    print(colored(f' Entire Accuracy: {pre_acc.shape}', 'cyan'))


if __name__ == "__main__":
    main()


