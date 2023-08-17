import argparse
import os
#model
import random

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
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored

# from acquisition_functions import uniform, max_entropy, bald, var_ratios, mean_std,cross_entropy

from models.models import ClusteringModel

from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from sampler import SubsetSequentialSampler
def plot_results(data):
    """Plot results histogram using matplotlib"""
    # sns.set()
    plt.figure()
    plt.subplot(111)
    plt.ylabel("acc")
    plt.xlabel("number")
    x_major_locator = MultipleLocator(20)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.1)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(-0.5, len(data)+5)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0.2, 0.6)
    # plt.xticks(a[::0.01])
    # plt.yticks(b[::10])
    # for key in data.keys():
        # data[key] = gaussian_filter1d(data[key], sigma=0.9) # for smoother graph
    plt.plot(data, label='cross')
    plt.savefig('./results/cifar-100/cross-3s.png')
    # plt.close()
    # plt.legend()
    # plt.show()
class ActiveModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ActiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert (isinstance(self.nheads, int))
        assert (self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])
        # self.Contrastive_head = nn.Linear(self.backbone_dim, 128)
        # self.head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])
        # self.head = nn.ModuleList([nn.Linear(nclusters, nclusters),
        #                            nn.Linear(nclusters, nclusters),
        #                            nn.Linear(nclusters, nclusters)
        #                            ])
    def feature(self,x,):
        features = self.backbone(x)

        return features


    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]
            out = out[0]
            return out,features

class MLP(nn.Module):
    def __init__(self,dim,nclusters):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(dim,256)
        self.layer2 = nn.Linear(256,256)
        self.layer3 = nn.Linear(256,nclusters)

    def forward(self, x) :
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out = self.layer3(out2)
        return out



def AcquisitionFunction(p,data,subset,device,num_class,n_query):
    num = int(n_query/num_class)
    # print(num)

    model = get_model(p, p['scan_model'])
    # print(p['scan_model'])

    model = model.to(device)
    # print(model.device)
    # print(model)
    print(colored(f'yu xun lian model:{model}','blue'))
    pretrain_path = p['selflabel_model']
    # state = torch.load(pretrain_path, map_location='cpu')

    print(colored('Retrieve model', 'blue'))

    subset_loader = DataLoader(data, batch_size=128,
                                    sampler=SubsetSequentialSampler(subset),
                                    pin_memory=False, drop_last=True)
    # model.load_state_dict(state, strict=True)

    # torch.backends.cudnn.enabled = False
    # print(model.device)
    # with torch.cuda.device(0):
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
        # print(cluster[i][0][:10])
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
    # print(query_set )
    # print(remainSet)
    # print(subset)
    return query_set,remainSet

def feature(device,model, subset,data,n_query :int = 1000):
    model.eval()


    subset_loader = DataLoader(data, batch_size=128,
                                    sampler=SubsetSequentialSampler(subset),
                                    pin_memory=True, drop_last=True)

    output_w= torch.tensor([]).to(device)
    output_s = torch.tensor([]).to(device)

    # softmax = nn.Softmax(dim=1)

    for i in subset_loader:
        with torch.no_grad():
            input = i['image']
            input = input.to(device)

            input_augmented = i['image_augmented']
            input_augmented = input_augmented.to(device)

            output_batch = model.feature(input)
            # output_batch = softmax(output_batch)
            output_w = torch.cat((output_w, output_batch), 0)

            output_augmented_batch = model.feature(input_augmented)
            # output_augmented_batch = softmax(output_augmented_batch)
            output_s = torch.cat((output_s, output_augmented_batch), 0)


    # output_w = output_w.cpu().numpy()
    # output_s = output_s.cpu().numpy()
    cos_sim = F.cosine_similarity(output_w, output_s, dim=1)
    cos_sim = cos_sim.cpu().numpy()


    idx = (cos_sim).argsort()[:n_query]

    query_set = list(np.array(subset)[idx])
    remainSet = set(np.array(subset)) - set(np.array(subset)[idx])
    remainSet = list(remainSet)

    return query_set,remainSet


def cross_entropy(device,model, subset,data,n_query :int = 1000):
    model.eval()


    subset_loader = DataLoader(data, batch_size=128,
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

    # H = np.mean((output_w-output_s)**2,axis=1)
    # H = np.linalg.norm(output_w-output_s,axis=1)
    # print(H.shape)
    #
    # idx = (-H).argsort()[:n_query]



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

def uniform(device,model, subset,data,n_query :int = 1000):

    query_set = np.random.choice(np.array(subset),size=n_query,replace = False)
    remainSet = set(np.array(subset)) - set(query_set)
    query_set = list(query_set)
    remainSet = list(remainSet)

    return query_set, remainSet

def visual(model,set,reset,data,device):
    model.eval()

    set_loader = DataLoader(data, batch_size=128,
                               sampler=SubsetSequentialSampler(set),
                               pin_memory=True, drop_last=False)
    reset_loader = DataLoader(data, batch_size=128,
                               sampler=SubsetSequentialSampler(reset),
                               pin_memory=True, drop_last=False)

    # classes =  [[] for i in range(11)]
    classes = dict()
    for j in range(10):
        classes[str(j)] = torch.tensor([]).to(device)
    output_w = torch.tensor([]).to(device)
    output_s = torch.tensor([]).to(device)

    # softmax = nn.Softmax(dim=1)
    for i in set_loader:
        with torch.no_grad():
            input = i['image']
            input = input.to(device)

            # input_augmented = i['image_augmented']
            # input_augmented = input_augmented.to(device)

            # target = i['target']
            # target = target.detach().cpu().numpy()

            output_batch,feas = model(input)
            output_s = torch.cat((output_s, output_batch), 0)

    classes['11'] = output_s

    for i in reset_loader:
        with torch.no_grad():
            input = i['image']
            input = input.to(device)

            # input_augmented = i['image_augmented']
            # input_augmented = input_augmented.to(device)

            target = i['target']
            target = target.detach().cpu().numpy()

            #按模型预测结果
            output_batch, fea = model(input)
            index = torch.argmax(output_batch, dim=1)
            index = index.detach().cpu().numpy()


            for k in range(10) :
                # index = torch.argmax(output_batch,dim=1)
                suo_yin = list(np.where(index == k))

                # suo_yin = list(np.where(target == k))
                # output_batch,fe = model(input[suo_yin])
                output_w = classes[str(k)]

                output_w = torch.cat((output_w, output_batch[suo_yin]), 0)
                classes[str(k)] = output_w
    # import ipdb
    # ipdb.set_trace()
    from sklearn.manifold import TSNE
    plt.figure(dpi=300,figsize=(5,5))
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])
    # x = np.array([])
    all_w = torch.tensor([]).to(device)
    # x_emd = PCA().fit_transform(x)
    for key in classes.keys():
        y = classes[key]
        all_w=torch.cat((all_w, y), 0)
        # x = np.concatenate((x, y))
    x = all_w.detach().cpu().numpy()

    x_emd = TSNE(n_components=2, learning_rate='auto', init='pca',early_exaggeration=1,perplexity=3,n_iter=2000).fit_transform(x)
    x_min, x_max = x_emd.min(0), x_emd.max(0)
    x_emd = (x_emd - x_min) / (x_max - x_min)
    leng = 0

    for key in classes.keys():

        if key == '11':
            plt.scatter(x_emd[leng:leng+len(classes[key]), 0], x_emd[leng:leng+len(classes[key]), 1],s=10,c='k',marker='x')
        else:

            plt.scatter(x_emd[leng:leng+500, 0], x_emd[leng:leng+500, 1],s=70)
        leng += len(classes[key])

    plt.savefig('./results/cifar-10/cifar20R.png',bbox_inches='tight')
    plt.close()


    # output_augmented_batch = model.feature(input_augmented)
                # # output_augmented_batch = softmax(output_augmented_batch)
                # output_s = torch.cat((output_s, output_augmented_batch), 0)



def activate_train(model,epoch,train_dataset,val_dataset,device):
    model.train()
    from torch.utils.data import DataLoader, random_split
    import torch.optim.lr_scheduler as lr_scheduler
    from losses.losses import SimCLRLoss
    is_c_loss = True

    if is_c_loss == True:
        # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1,
        #                            momentum=0.9, weight_decay=5e-4)
        # sched_backbone = lr_scheduler.MultiStepLR(optimizer, milestones=[160, 240])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0001)
        sched_backbone = lr_scheduler.MultiStepLR(optimizer, milestones=[120,180])  # cifar10[5,20]
        # c_loss = SimCLRLoss(0.1)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.0001)
        sched_backbone = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 20])#cifar10[5,20]
    # print(colored(optimizer,'blue'))
    loss_function = nn.CrossEntropyLoss(reduction='mean')
    c_loss = SimCLRLoss(0.1)




    from tqdm import tqdm
    for k in range(epoch):

        for i in tqdm(train_dataset,leave=False):
            images = i['image']
            images_augmented = i['image_augmented']
            target = i['target']

            images = images.to(device)
            images_augmented = images_augmented.to(device)
            target = target.to(device)


            #对比损失
            # b, c, h, w = images.size()
            # input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
            # input_ = input_.view(-1, c, h, w)
            # # print(input_.shape)
            # output,_ = model(input_)
            # output = output.view(b, 2, -1)
            # contrastive_loss = c_loss(output)
                # import ipdb
                # ipdb.set_trace()
            output_w,_ = model(images)
            output_s,_ = model(images_augmented)

            loss_y = torch.mean((output_w-output_s)**2)
            loss_w = loss_function(output_w, target)
            loss_s = loss_function(output_s, target)
            loss = loss_w+ loss_s + loss_y






            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # if is_c_loss == True:
        sched_backbone.step()
    print(colored('-->Finished.','red'))


def evaluate_accuracy_ac(model,data,device):

    # import ipdb
    # ipdb.set_trace()

    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_sum, n = 0.0, 0
    for i in data:
        image = i['image'].to(device)
        target = i['target'].to(device)

        # for image in data:
        with torch.no_grad():

            # x =x.unsqueeze(0)
            pred,_ = model(image)
            pred = pred.argmax(dim=1)
            acc_sum  +=  pred.eq(target).sum().item()
        n += target.shape[0]
    return acc_sum / n



def main():

    #model
    p = create_config('./configs/env.yml', './configs/selflabel/selflabel_cifar100.yml', 1, 100)
    print(colored(p, 'red'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from models.resnet_cifar import resnet18
    backbone = resnet18()
    model = ActiveModel(backbone,p['num_classes'])

    model = model.to(device)
    pretrain_path = p['selflabel_model']
    # state = torch.load(pretrain_path, map_location='cpu')
    # state = torch.load(pretrain_path)
    # model.load_state_dict(state,strict=False)
    # model = torch.nn.DataParallel(model)
    print(model)


    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True



    # Dataset
    print(colored('Retrieve dataset', 'blue'))

    # Transforms
    import torchvision.transforms as transforms
    from data.augment import Augment, Cutout
    from utils.collate import collate_custom
    strong_transforms =get_train_transformations(p)
    val_transforms = get_val_transformations(p)

    # val_t =transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(**p['transformation_kwargs']['normalize'])])

    print(colored(f'strong_transforms:{strong_transforms}','blue'))


    #data

    train_dataset_ = get_train_dataset(p, {'standard': val_transforms, 'augment': strong_transforms},
                                      split='train', to_augmented_dataset=True)
    import random
    ADDNUM = 10000
    n_dataset = len(train_dataset_)
    label_set = []
    unlabel_set = [x for x in range(n_dataset)]
    random.shuffle(unlabel_set)
    subset = unlabel_set[:ADDNUM]

    val_dataset_ = get_val_dataset(p, val_transforms)
    val_dataset = torch.utils.data.DataLoader(val_dataset_ , num_workers=8,
                                batch_size=128, pin_memory=False, collate_fn=collate_custom,
                                drop_last=False, shuffle=False)
    print(colored('Train samples %d - Val samples %d' % (len(train_dataset_), len(val_dataset)), 'yellow'))


    #查找启动数据1000张
    #init data
    print(colored('query init data','blue'))
    query_set,remainSet = AcquisitionFunction(p, train_dataset_,subset,device,p['num_classes'],1000)

    label_set += query_set
    unlabel_set = unlabel_set[ADDNUM:] + remainSet
    label_set_loader = DataLoader(train_dataset_, batch_size=128,
                               sampler=SubsetRandomSampler(label_set),
                               pin_memory=True, drop_last=True)
    # print(type(index_start))
    print(colored(len(unlabel_set),'yellow'))
    print(colored(f'the number of init_data:{len(label_set_loader)}','cyan'))

    print(colored(f'看一眼数据对不对{len(label_set)}','red'))
    pre_acc = []
    print(colored('start active with uncertain ', 'yellow'))
    epoch = 10

    for i in range(epoch):


        print(colored(f'the number of train data {i+1} lun: {len(label_set)} ', 'cyan'))
        activate_train(model,200,label_set_loader,val_dataset,device)
        acc = evaluate_accuracy_ac(model,val_dataset,device)

        print(colored(f"active train lun ci : {i+1} acc: {acc}", 'yellow'))

        #query next iter data
        if i < (epoch - 0) :
            random.shuffle(unlabel_set)
            subset = unlabel_set[:ADDNUM]
            query_set, remainSet = cross_entropy(device, model, subset,train_dataset_, n_query=1000)
            # query_set, remainSet = feature(device, model, subset, train_dataset_, n_query=1000)
            # query_set, remainSet = uniform(device, model, subset,train_dataset_, n_query=1000)
            print(colored(f'shu ju query_set: {len(query_set)}, remainSet: {len(remainSet)}','blue'))
            label_set += query_set
            unlabel_set = unlabel_set[ADDNUM:] + remainSet
            label_set_loader = DataLoader(train_dataset_, batch_size=128,
                                          sampler=SubsetRandomSampler(label_set),
                                          pin_memory=True, drop_last=True)
            print(colored(f'the number of unlabel data {i+1} lun: {len(unlabel_set)} ', 'red'))



        pre_acc.append(acc)
        print(colored(f" acc of len:{len(pre_acc)} ",'yellow'))
    # visual(model, query_set, remainSet, train_dataset_, device)
    pre_acc = np.array(pre_acc)
    np.save('./results/cifar-100/cifar100-U(c).npy', pre_acc)
    print(colored(f'zhun que lu {pre_acc.shape}', 'cyan'))

    plot_results(list(pre_acc))



if __name__ == "__main__":

    main()


