import time
import tqdm
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.autograd as autograd

import torchvision.models as models

from agents.base import BaseAgent
from graphs.models.vgg import *
from prune.channel import *
from datasets.imagenet import *
from datasets.cifar100 import *

from utils.metrics import AverageMeter, cls_accuracy
from utils.misc import timeit, print_cuda_statistics
from math import cos, pi

cudnn.benchmark = True


class VGG_BN_cifar(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # set device
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = self.config.seed
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)

            self.logger.info("Program will run on *****GPU-CUDA*****\n")
            # print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        self.num_classes = self.config.num_classes

        self.model = None       # original model graph, loss function, optimizer, learning scheduler
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None

        self.data_loader = Cifar100DataLoader(config=self.config)   # data loader
        self.sub_data_loader = None                                 # sub data loader for sub task

        self.current_epoch = 0      # info for train
        self.current_iteration = 0
        self.best_valid_acc = 0

        self.cls_i = None
        self.channel_importance = dict()

        self.all_list = list()

        self.named_modules_list = dict()
        self.named_conv_list = dict()

        self.original_conv_output = dict()

        self.init_graph()

    def init_graph(self, pretrained=True, init_channel_importance=True):     # 모델 그래프와 정보를 초기화
        # set model graph & information holder
        self.model = vgg16(input_shape=self.config.img_size, num_classes=self.config.num_classes, batch_norm=True)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005,
                                         nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.milestones,
                                                        gamma=self.config.gamma)
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

        self.cls_i = None
        self.channel_importance = dict()

        self.named_modules_idx_list = dict()
        self.named_modules_list = dict()
        self.named_conv_list = dict()
        self.named_conv_idx_list = dict()

        self.original_conv_output = dict()

        self.stayed_channels = dict()

        i = 0
        for idx, m in enumerate(self.model.features):
            if isinstance(m, torch.nn.Conv2d):
                self.named_modules_idx_list['{}.conv'.format(i)] = idx
                self.named_modules_list['{}.conv'.format(i)] = m
                self.named_conv_idx_list['{}.conv'.format(i)] = idx
                self.named_conv_list['{}.conv'.format(i)] = m
            elif isinstance(m, torch.nn.BatchNorm2d):
                self.named_modules_idx_list['{}.bn'.format(i)] = idx
                self.named_modules_list['{}.bn'.format(i)] = m
                i += 1

        if init_channel_importance is True:
            self.channel_importance = dict()

    def set_subtask(self, cls_i):
        # cls_i should have list type
        query_task = []
        for task in cls_i:
            query_task.append(task)
        self.cls_i = query_task

        self.sub_data_loader = SpecializedCifar100DataLoader(self.config, self.cls_i)

    def load_checkpoint(self, file_path="checkpoint.pth", only_weight=False):
        """
        Latest checkpoint loader
        :param file_path: str, path of the checkpoint file
        :param only_weight: bool, load only weight or all training state
        :return:
        """
        try:
            self.logger.info("Loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path)
            if only_weight:
                self.model.load_state_dict(checkpoint)
                self.logger.info("Checkpoint loaded successfully\n")
            else:
                self.current_epoch = checkpoint['epoch']
                self.current_iteration = checkpoint['iteration']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.logger.info("Checkpoint loaded successfully at (epoch {}) at (iteration {})\n"
                                 .format(checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists")

    def save_checkpoint(self, file_name="checkpoint.pth", is_best=False):
        pass

    def record_channel_importance(self):
        def save_grad(idx):
            def hook(grad):
                grads[idx] = grad
            return hook

        for n, m in self.named_conv_list.items():
            self.channel_importance[n] = torch.zeros(m.out_channels)

        for inputs, labels in self.data_loader.train_loader:
            num_batch = inputs.size(0)
            outputs, grads = {}, {}
            if self.cuda:
                inputs = inputs.cuda(non_blocking=self.config.async_loading)
                inputs.requires_grad = True

            x = inputs
            i = 0
            for m in self.model.features:
                x = m(x)
                if isinstance(m, torch.nn.ReLU):
                    outputs['{}.conv'.format(i)] = x
                    outputs['{}.conv'.format(i)].register_hook(save_grad('{}.conv'.format(i)))
                    i += 1
            else:
                x = x.view(num_batch, -1)

            x = self.model.classifier(x)

            y_hat = x
            y_hat[:, self.cls_i].backward(gradient=torch.ones_like(y_hat[:, self.cls_i]))

            self.cal_importance(grads, outputs)

    def cal_importance(self, grads_list, outputs_list):
        for n, m in self.named_conv_list.items():
            if isinstance(m, torch.nn.Conv2d):
                grad = grads_list[n]
                output = outputs_list[n]
                importance = (grad).mean(dim=(2, 3))
                total_importance = torch.abs(importance).sum(dim=0)
                self.channel_importance[n] += total_importance.data.cpu()

    def record_conv_output(self, inputs):
        x = inputs
        i = 0
        for m in self.model.features:
            x = m(x)
            if isinstance(m, torch.nn.Conv2d):
                self.original_conv_output['{}.conv'.format(i)] = x.data
                i += 1

    def run(self):
        pass

    @timeit
    def compress(self, method='gradient', k=0.5):
        if method == "first_k":
            for i, m in enumerate(list(self.named_conv_list.values())[:-1]):    # 마지막 레이어 전까지
                bn = self.named_modules_list[str(i) + '.bn']
                if str(i + 1) + '.conv' in self.named_conv_list:
                    next_m = self.named_modules_list[str(i + 1) + '.conv']
                else:
                    next_m = self.model.classifier[0]
                indices_stayed = [i for i in range(math.floor(m.out_channels * k))]
                module_surgery(m, bn, next_m, indices_stayed)
                self.stayed_channels[str(i) + '.conv'] = set(indices_stayed)
            return

        inputs, _ = next(iter(self.data_loader.train_loader))
        if self.cuda:
            inputs = inputs.cuda(non_blocking=self.config.async_loading)
        self.record_conv_output(inputs)
        
        if method == 'manual':  # 미리 저장된 레이어당 채널 번호로 프루닝
            for i, m in enumerate(self.named_conv_list.values()):
                if isinstance(m, torch.nn.Conv2d):
                    bn = self.named_modules_list[str(i) + '.bn']
                    if str(i + 1) + '.conv' in self.named_conv_list:
                        next_m = self.named_modules_list[str(i + 1) + '.conv']
                    else:
                        next_m = self.model.classifier[0]

                    indices_stayed = list(self.stayed_channels[str(i) + '.conv'])
                    module_surgery(m, bn, next_m, indices_stayed)
                    if not isinstance(next_m, torch.nn.Linear):
                        next_output_features = self.original_conv_output[str(i + 1) + '.conv']
                        next_m_idx = self.named_conv_idx_list[str(i + 1) + '.conv']
                        pruned_next_inputs_features = self.model.features[:next_m_idx](inputs)
                        weight_reconstruction(next_m, pruned_next_inputs_features, next_output_features, use_gpu=self.cuda)
                self.stayed_channels[str(i) + '.conv1'] = set(indices_stayed)
        elif method == 'gradient':
            if not self.channel_importance:
                self.record_channel_importance()
            for i, m in enumerate(self.named_conv_list.values()):
                if isinstance(m, torch.nn.Conv2d):
                    bn = self.named_modules_list[str(i) + '.bn']
                    if str(i + 1) + '.conv' in self.named_conv_list:
                        next_m = self.named_modules_list[str(i + 1) + '.conv']
                    else:
                        next_m = self.model.classifier[0]
                    channel_importance = self.channel_importance[str(i) + '.conv']
                    channel_importance = channel_importance / channel_importance.sum()
                    threshold = k / channel_importance.size(0)
                    indices_stayed = [i for i in range(len(channel_importance)) if channel_importance[i] > threshold]
                    module_surgery(m, bn, next_m, indices_stayed)
                    if not isinstance(next_m, torch.nn.Linear):
                        next_output_features = self.original_conv_output[str(i + 1) + '.conv']
                        next_m_idx = self.named_conv_idx_list[str(i + 1) + '.conv']
                        pruned_next_inputs_features = self.model.features[:next_m_idx](inputs)
                        weight_reconstruction(next_m, pruned_next_inputs_features, next_output_features, use_gpu=self.cuda)
                self.stayed_channels[str(i) + '.conv'] = set(indices_stayed)

        elif method == 'max_output': # NO weight reconstuction
            for i, m in enumerate(list(self.named_conv_list.values())[:-1]):    # 마지막 레이어 전까지
                if isinstance(m, torch.nn.Conv2d):
                    bn = self.named_modules_list[str(i) + '.bn']
                    if str(i + 1) + '.conv' in self.named_conv_list:
                        next_m = self.named_modules_list[str(i + 1) + '.conv']
                    else:
                        next_m = self.model.classifier[0]
                    num_channel = int(k * m.out_channels)
                    channel = self.original_conv_output[str(i)+'.conv']
                    channel_vec = channel.view(channel.size()[1], -1)
                    channel_norm = torch.norm(channel_vec,2,1)
                    indices_stayed = torch.argsort(channel_norm, descending=True)[:num_channel]    # 가장 큰 채널들의 index를 리턴
                    module_surgery(m, bn, next_m, indices_stayed)


        elif method == 'greedy':
            for i, m in enumerate(list(self.named_conv_list.values())[:-1]):    # 마지막 레이어 전까지
                if isinstance(m, torch.nn.Conv2d):
                    next_m_idx = self.named_modules_idx_list[str(i + 1) + '.conv']
                    bn, next_m = self.named_modules_list[str(i) + '.bn'], self.named_modules_list[str(i + 1) + '.conv']
                    next_input_features = self.model.features[:next_m_idx](inputs)
                    indices_stayed, indices_pruned = channel_selection(next_input_features, next_m, sparsity=(1. - k),
                                                                       method='greedy')
                    module_surgery(m, bn, next_m, indices_stayed)

                    next_output_features = self.original_conv_output[str(i + 1) + '.conv']
                    next_m_idx = self.named_conv_idx_list[str(i + 1) + '.conv']
                    pruned_next_inputs_features = self.model.features[:next_m_idx](inputs)
                    weight_reconstruction(next_m, pruned_next_inputs_features, next_output_features, use_gpu=self.cuda)
                    self.stayed_channels[str(i) + '.conv'] = set(indices_stayed)

        elif method == 'lasso':
            for i, m in enumerate(list(self.named_conv_list.values())[:-1]):    # 마지막 레이어 전까지
                if isinstance(m, torch.nn.Conv2d):
                    next_m_idx = self.named_modules_idx_list[str(i + 1) + '.conv']
                    bn, next_m = self.named_modules_list[str(i) + '.bn'], self.named_modules_list[str(i + 1) + '.conv']
                    next_input_features = self.model.features[:next_m_idx](inputs)
                    indices_stayed, indices_pruned = channel_selection(next_input_features, next_m, sparsity=(1. - k),
                                                                       method='lasso')
                    module_surgery(m, bn, next_m, indices_stayed)

                    next_output_features = self.original_conv_output[str(i + 1) + '.conv']
                    next_m_idx = self.named_conv_idx_list[str(i + 1) + '.conv']
                    pruned_next_inputs_features = self.model.features[:next_m_idx](inputs)
                    weight_reconstruction(next_m, pruned_next_inputs_features, next_output_features, use_gpu=self.cuda)
                    self.stayed_channels[str(i) + '.conv'] = set(indices_stayed)

    def adjust_learning_rate(self,optimizer, epoch, iteration, num_iter):
        warmup_epoch = 5
        warmup_iter = warmup_epoch * num_iter
        current_iter = iteration + epoch * num_iter
        max_iter = 100 * num_iter

        lr = 0.1 * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2

        if epoch < warmup_epoch:
            lr = 0.1* current_iter / warmup_iter

        if iteration == 0:
            print('current learning rate:{0}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    @timeit
    def train(self, specializing=False, freeze_conv=False):
        """
        Main training function, with per-epoch model saving
        :return:
        """
        if freeze_conv:
            for param in self.model.features.parameters():
                param.requires_grad = False

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005,
                                         nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.milestones,
                                                        gamma=self.config.gamma)
        self.model.to(self.device)

        history = []
        for epoch in range(self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch(specializing)

            if specializing:
                sub_valid_acc = []
                sub_valid_acc.append(self.validate(specializing))
                valid_acc = np.mean(sub_valid_acc)
            else:
                valid_acc = self.validate(specializing)
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
            self.save_checkpoint(is_best=is_best)

            history.append(valid_acc)
            self.scheduler.step(valid_acc)

        if freeze_conv:
            for param in self.model.features.parameters():
                param.requires_grad = True

        return self.best_valid_acc, history

    def train_one_epoch(self, specializing=False):
        """
        One epoch training function
        :return:
        """
        if specializing:
            tqdm_batch = tqdm.tqdm(self.sub_data_loader.binary_train_loader,
                                   total=self.sub_data_loader.binary_train_iterations,
                                   desc="Epoch-{}-".format(self.current_epoch))
        else:
            tqdm_batch = tqdm.tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                                   desc="Epoch-{}-".format(self.current_epoch))

        self.model.train()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        current_batch = 0
        for i,(x, y) in enumerate(tqdm_batch):
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(non_blocking=self.config.async_loading)

            self.optimizer.zero_grad()
            self.adjust_learning_rate(self.optimizer, self.current_epoch, i, self.data_loader.train_iterations)

            pred = self.model(x)
            cur_loss = self.loss_fn(pred, y)


            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            cur_loss.backward()
            self.optimizer.step()

            if specializing:
                top1 = cls_accuracy(pred.data, y.data)
                top1_acc.update(top1[0].item(), x.size(0))
            else:
                top1, top5 = cls_accuracy(pred.data, y.data, topk=(1, 5))
                top1_acc.update(top1.item(), x.size(0))
                top5_acc.update(top5.item(), x.size(0))

            epoch_loss.update(cur_loss.item())

            self.current_iteration += 1
            current_batch += 1

        tqdm_batch.close()

        print("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(epoch_loss.val) +
                         "\tTop1 Acc: " + str(top1_acc.val))

    def validate(self, specializing=False):
        """
        One epoch validation
        :return:
        """
        if specializing:
            tqdm_batch = tqdm.tqdm(self.sub_data_loader.binary_valid_loader,
                                   total=self.sub_data_loader.binary_valid_iterations,
                                   desc="Epoch-{}-".format(self.current_epoch))
        else:
            tqdm_batch = tqdm.tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                                   desc="Valiation at -{}-".format(self.current_epoch))

        self.model.eval()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(non_blocking=self.config.async_loading)

            # model
            pred = self.model(x)
            # loss
            cur_loss = self.loss_fn(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during validation...')

            if specializing:
                top1 = cls_accuracy(pred.data, y.data)
                top1_acc.update(top1[0].item(), x.size(0))
            else:
                top1, top5 = cls_accuracy(pred.data, y.data, topk=(1, 5))
                top1_acc.update(top1.item(), x.size(0))
                top5_acc.update(top5.item(), x.size(0))

            epoch_loss.update(cur_loss.item())

        print("Validation results at epoch-" + str(self.current_epoch) + " | " + "loss: " +
                         str(epoch_loss.avg) + "\tTop1 Acc: " + str(top1_acc.val))

        tqdm_batch.close()

        return top1_acc.avg

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """