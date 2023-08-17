'''
GCN Active Learning
'''

# Python
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import argparse
# Custom
import models.resnet as resnet
from models.resnet import vgg11
from models.query_models import LossNet
from train_test import train, test
from load_dataset import load_dataset
from selection_methods import query_samples
from config import *

from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--lambda_loss", type=float, default=1.2,
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s", "--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n", "--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r", "--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d", "--dataset", type=str, default="cifar10",
                    help="")
parser.add_argument("-e", "--no_of_epochs", type=int, default=20,
                    help="Number of epochs for the active learner")
parser.add_argument("-m", "--method_type", type=str, default="CoreGCN",
                    help="")
parser.add_argument("-c", "--cycles", type=int, default=10,
                    help="Number of active learning cycles")
parser.add_argument("-t", "--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("-p", "--path", type=str, default='./results/cifar-10/',
                    help='Path of save acc')

args = parser.parse_args()


class ClusteringModel(nn.Module):
	def __init__(self, backbone, nclusters, nheads=1):
		super(ClusteringModel, self).__init__()
		self.backbone = backbone['backbone']
		self.backbone_dim = backbone['dim']
		self.nheads = nheads
		assert (isinstance(self.nheads, int))
		assert (self.nheads > 0)
		self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])

	def forward(self, x, forward_pass='default'):
		if forward_pass == 'default':
			features= self.backbone(x)
			out = [cluster_head(features) for cluster_head in self.cluster_head]

		elif forward_pass == 'backbone':
			out = self.backbone(x)

		elif forward_pass == 'head':
			out = [cluster_head(x) for cluster_head in self.cluster_head]

		elif forward_pass == 'return_all':
			features = self.backbone(x)
			out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}

		else:
			raise ValueError('Invalid forward pass {}'.format(forward_pass))

		return out

def AcquisitionFunction(data,subset,device,num_class,n_query):
    num = int(n_query/num_class)
    from models.resnet_cifar import resnet18
    backbone = resnet18()
    # print(num)

    model = ClusteringModel(backbone,10,1)
    # print(model)
    state = torch.load('./cifar10/scan/model_seed1_clusters10.pth.tar', map_location='cpu')
    # print(state)
    model_state = state['model']
    all_heads = [k for k in model_state.keys() if 'cluster_head' in k]
    best_head_weight = model_state['cluster_head.%d.weight' % (state['head'])]
    best_head_bias = model_state['cluster_head.%d.bias' % (state['head'])]
    for k in all_heads:
	    model_state.pop(k)

    model_state['cluster_head.0.weight'] = best_head_weight
    model_state['cluster_head.0.bias'] = best_head_bias

    # print(state)
    missing = model.load_state_dict(model_state, strict=True)
    print(missing)
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        model = model.cuda()
	# model.eval()
    # print(model.device)
    # print(model)
    # print(colored(f'yu xun lian model:{model}','blue'))
    # pretrain_path = p['selflabel_model']
    # state = torch.load(pretrain_path, map_location='cpu')

    # print(colored('Retrieve model', 'blue'))
    from data.sampler import SubsetSequentialSampler

    subset_loader = DataLoader(data, batch_size=128,
                                    sampler=SubsetSequentialSampler(subset),
                                    pin_memory=False, drop_last=True)
    # model.load_state_dict(state, strict=True)

    # torch.backends.cudnn.enabled = False
    # print(model.device)
    # with torch.cuda.device(0):
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        output = torch.tensor([]).cuda()
    softmax = nn.Softmax(dim=1)
    # import ipdb
    # ipdb.set_trace()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
	    for i,j in subset_loader:
	        with torch.no_grad():
	            input = i
	            input = input.cuda()
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
    print(len(query_set) )
    # print(remainSet)
    # print(subset)
    return query_set,remainSet
##
# Main
if __name__ == '__main__':

	method = args.method_type
	methods = ['Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss', 'VAAL']
	datasets = ['cifar10', 'cifar100', 'fashionmnist', 'svhn']
	assert method in methods, 'No method %s! Try options %s' % (method, methods)
	assert args.dataset in datasets, 'No dataset %s! Try options %s' % (args.dataset, datasets)
	'''
	method_type: 'Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL'
	'''
	results = open(
		'results_' + str(args.method_type) + "_" + args.dataset + '_main' + str(args.cycles) + str(args.total) + '.txt',
		'w')
	print(colored("Dataset: %s" % args.dataset, 'blue'))
	print(colored("Method type:%s" % method, 'cyan'))
	if args.total:
		TRIALS = 1
		CYCLES = 1
	else:
		CYCLES = args.cycles
	for trial in range(TRIALS):

		# Load training and testing dataset
		data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args.dataset)
		# Don't predefine budget size. Configure it in the config.py: ADDENDUM = adden
		NUM_TRAIN = no_train
		indices = list(range(NUM_TRAIN))
		random.shuffle(indices)

		if args.total:
			labeled_set = indices
		else:
			device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
			subset = indices[:SUBSET]
			query_set,remainSet = AcquisitionFunction(data_train, subset, device, 10, 1000)
			labeled_set = query_set
			# unlabel_set = unlabel_set[ADDNUM:] + remainSet
			# labeled_set = indices[:ADDENDUM]
			unlabeled_set = [x for x in indices if x not in labeled_set]

		train_loader = DataLoader(data_train, batch_size=BATCH,
		                          sampler=SubsetRandomSampler(labeled_set),
		                          pin_memory=True, drop_last=True)
		test_loader = DataLoader(data_test, batch_size=BATCH)
		dataloaders = {'train': train_loader, 'test': test_loader}
		pre_acc = []


		# Randomly sample 10000 unlabeled data points
		if not args.total:
			random.shuffle(unlabeled_set)
			subset = unlabeled_set[:SUBSET]

		# Model - create new instance for every cycle so that it resets
		with torch.cuda.device(CUDA_VISIBLE_DEVICES):
			if args.dataset == "fashionmnist":
				resnet18 = resnet.ResNet18fm(num_classes=NO_CLASSES).cuda()
			else:
				# resnet18    = vgg11().cuda()
				resnet18 = resnet.ResNet18(num_classes=NO_CLASSES).cuda()
				net = resnet.Rnet(resnet18)
			if method == 'lloss':
				if args.dataset == "fashionmnist":
					loss_module = LossNet(feature_sizes=[28, 14, 7, 4], num_channels=[64, 128, 256, 512]).cuda()
				else:
					loss_module = LossNet().cuda()

		state = torch.load('./model_seed1_clusters10.pth.tar', map_location='cpu')
		# print(colored(resnet18,'blue'))

		missing = net.load_state_dict(state, strict=False)
		print(colored(missing, 'yellow'))

		models = {'backbone': net}
		if method == 'lloss':
			models = {'backbone': net, 'module': loss_module}
		torch.backends.cudnn.benchmark = True


		for cycle in range(CYCLES):



			# Loss, criterion and scheduler (re)initialization
			criterion = nn.CrossEntropyLoss(reduction='none')
			# optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
			#                            momentum=MOMENTUM, weight_decay=WDECAY)
			optim_backbone = optim.Adam(models['backbone'].parameters(), lr=0.0001,  weight_decay=0.0001)

			sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=[5,20])
			optimizers = {'backbone': optim_backbone}
			schedulers = {'backbone': sched_backbone}
			if method == 'lloss':
				optim_module = optim.SGD(models['module'].parameters(), lr=LR,
				                         momentum=MOMENTUM, weight_decay=WDECAY)
				sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=[5,20])
				optimizers = {'backbone': optim_backbone, 'module': optim_module}
				schedulers = {'backbone': sched_backbone, 'module': sched_module}

			# Training and testing
			train(models, method, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs, EPOCHL)
			acc = test(models, EPOCH, method, dataloaders, mode='test')
			print(colored(
				'Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
				                                                                      CYCLES, len(labeled_set), acc),
				'yellow'))
			np.array([method, trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
			results.write("\n")
			pre_acc.append(acc)

			if cycle == (CYCLES - 1):
				# Reached final training cycle
				print("Finished.")
				break
			# Get the indices of the unlabeled samples to train on next cycle
			arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args)

			# Update the labeled dataset and the unlabeled dataset, respectively
			labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
			listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy())
			unlabeled_set = listd + unlabeled_set[SUBSET:]
			print(len(labeled_set), min(labeled_set), max(labeled_set))
			# Create a new dataloader for the updated labeled dataset
			dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
			                                  sampler=SubsetRandomSampler(labeled_set),
			                                  pin_memory=True)
		path = args.path + str(method) + str(trial) + 'self' + '.npy'
		# savedir = os.path.join(path,str(method),str(trial))
		print(path)
		pre_acc = np.array(pre_acc)
		np.save(path, pre_acc)

	results.close()
