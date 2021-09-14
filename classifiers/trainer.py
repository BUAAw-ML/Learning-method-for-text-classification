import os
import shutil
import time
import random
from tqdm import tqdm

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet as tnt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import tensorflow as tf

import json

from datasets import DatasetForBert

from metrics.mltc_metric import *

tqdm.monitor_interval = 0


class Trainer(object):
    def __init__(self, state={}):

        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('train_bs') is None:
            self.state['train_bs'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('train_epochs') is None:
            self.state['train_epochs'] = 3

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()

        if self._state('print_freq') is None:
            self.state['print_freq'] = 0
        # best score
        self.state['best_score'] = {'map': 0., 'OP': 0., 'OR': 0., 'OF1': 0., 'CP': 0., 'CR': 0., 'CF1': 0., 'best_epoch': 0}

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def reset_best_score(self):
        self.state['best_score'] = {'map': 0., 'OP': 0., 'OR': 0., 'OF1': 0., 'CP': 0., 'CR': 0., 'CF1': 0., 'best_epoch': 0}

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=False):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=False):
        loss = self.state['meter_loss'].value()[0]
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=False):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=False):
        # record loss
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])


class MC_Trainer(Trainer): #training the model of multi-classification task
    def __init__(self, state):
        Trainer.__init__(self, state)

    def learning(self, model, trainset, testset):

        model.robust_tune_init("adamw", 0.00002, 0.001, 5000)

        # # TODO define optimizer
        random.shuffle(trainset)

        pbar = tqdm.tqdm(total=self.state['max_epochs'])
        for epoch in range(0, self.state['max_epochs']):

            correct_cnt = self.train(model, trainset)

            pbar.update(1)
            pbar.set_postfix({
                "training_acc": correct_cnt / len(trainset),
            })
        
        acc = self.validate(model)

        return acc


    def train(self, model, data):
        
        correct_cnt = 0
        for batch in range(0, len(data), self.state['train_bs']):
            batch_data = data[batch : batch + self.state['train_bs']]
            predict_list, loss = model.robust_tune_step(batch_data)
            for sample, predict in zip(batch_data, predict_list):
                if predict != sample["label"]: 
                    correct_cnt += 1
        
        return correct_cnt

    
    @torch.no_grad()
    def validate(self, model):
        
        acc = model.evaluate_classifier(self.state['eval_bs'])
        return acc

class MLC_Trainer(Trainer): #training the model of multi-label classification task
    def __init__(self, state):
        Trainer.__init__(self, state)
        self.state['ap_meter'] = AveragePrecisionMeter()

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, return_feature=False, return_penultimateLayer=False):

        target_var = self.state['target']
        ids, token_type_ids, attention_mask = self.state['input']
        print("on_forward")
        if self.state['use_gpu']:
            ids = ids.cuda(self.state['device_ids'][0])
            token_type_ids = token_type_ids.cuda(self.state['device_ids'][0])
            attention_mask = attention_mask.cuda(self.state['device_ids'][0])
            target_var = target_var.cuda(self.state['device_ids'][0])

        if return_feature:
            return model(ids, token_type_ids, attention_mask, return_feature=True)
        
        if return_penultimateLayer:
            return model(ids, token_type_ids, attention_mask, return_penultimateLayer=True)

        logits = model(ids, token_type_ids, attention_mask)

        self.state['output'] = logits

        if training:
            self.state['loss'] = criterion(logits, target_var)

            optimizer.zero_grad()
            self.state['loss'].backward()
            # nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], max_norm=10.0)
            optimizer.step()
        else:
            return self.state['output']

    def learning(self, model, trainset, testset, optimizer=None):

        # optionally resume from a checkpoint
        # if self._state('resume') is not None:
        #     if os.path.isfile(self.state['resume']):
        #         print("=> loading checkpoint '{}'".format(self.state['resume']))
        #         checkpoint = torch.load(self.state['resume'])
        #         self.state['start_epoch'] = checkpoint['epoch']
        #         self.state['best_score'] = checkpoint['best_score']
        #         model['Generator'].load_state_dict(checkpoint['state_dict-Generator'])
        #         model['Classifier'].load_state_dict(checkpoint['state_dict-Classifier'])
        #         print("=> loaded checkpoint '{}' (epoch {})"
        #               .format(self.state['evaluate'], checkpoint['epoch']))
        #     else:
        #         print("=> no checkpoint found at '{}'".format(self.state['resume']))

        # train_loader = torch.utils.data.DataLoader(
        # DatasetForBert(trainset, self.state["model_init"], self.state["train_bs"], self.state["label_num"]), batch_size=None, num_workers=1)

        # val_loader = torch.utils.data.DataLoader(
        # DatasetForBert(testset, self.state["model_init"], self.state["eval_bs"], self.state["label_num"]), batch_size=None, num_workers=1)
        #self.state['workers']
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=self.state['train_bs'], shuffle=True,
                                                   num_workers=self.state['workers'], collate_fn=self.state["dataset_object"].collate_fn)
        val_loader = torch.utils.data.DataLoader(testset,
                                                 batch_size=self.state['eval_bs'], shuffle=False,
                                                 num_workers=self.state['workers'], collate_fn=self.state["dataset_object"].collate_fn) #

        # TODO define optimizer
        optimizer = torch.optim.SGD(model.get_config_optim(self.state['learning_rate']),
                                    lr=self.state['learning_rate'],
                                    momentum=self.state['momentum'],
                                    weight_decay=self.state['weight_decay'])

        # define loss function (criterion)
        criterion = nn.BCELoss()#.MultiLabelSoftMarginLoss() #weight=torch.from_numpy(np.array(tag_weight)).float().cuda(0)

        if self.state['use_gpu']:
            # train_loader.pin_memory = True
            # val_loader.pin_memory = True
            # cudnn.benchmark = True
            model = model.cuda(self.state['device_ids'][0])
            criterion = criterion.cuda(self.state['device_ids'][0])

            # model = torch.nn.DataParallel(model, device_ids=self.state['device_ids'])

        # if self.state['evaluate']:
        #     self.state['epoch'] = self.state['start_epoch']
        #     self.validate(testset, model, criterion, self.state['epoch'])
        #     return


        pbar = tqdm(total=self.state['train_epochs'])

        for epoch in range(self.state['train_epochs']):
            
            print("learning")
            self.state['epoch'] = epoch
            # lr = self.adjust_learning_rate(optimizer)

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            score = self.validate(val_loader, model, criterion, epoch)
            if score["map"] > self.state['best_score']["map"]:
                self.state['best_score']['map'] = score['map']
            
            if score["OF1"] > self.state['best_score']["OF1"]:
                self.state['best_score']['OF1'] = score['OF1']
                self.state['best_score']['OP'] = score['OP']
                self.state['best_score']['OR'] = score['OR']
            
            if score["CF1"] > self.state['best_score']["CF1"]:
                self.state['best_score']['CF1'] = score['CF1']
                self.state['best_score']['CP'] = score['CP']
                self.state['best_score']['CR'] = score['CR']
                self.state['best_score']['best_epoch'] = epoch

 

                # remember best prec@1 and save checkpoint
                # self.save_checkpoint({
                #     'epoch': epoch + 1,
                #     # 'arch': self._state('arch'),
                #     'state_dict-Generator': model['Generator'].state_dict() if self.state['use_gpu'] else model['Generator'].state_dict(),
                #     'state_dict-Classifier': model['Classifier'].state_dict() if self.state['use_gpu'] else model['Classifier'].state_dict(),
                #     'best_score': self.state['best_score'],
                # }, is_best=1)

            # best_str = '**best_OF1** OF1={OF1:.3f} CF1={CF1:.3f} map={map:.3f}'.format( 
            #     OF1=self.state['best_score']['OF1'], 
            #     CF1=self.state['best_score']['CF1'],
            #     map=self.state['best_score']['map'])
            # print(best_str)
            # self.result_file.write(best_str + '\n')
            
            pbar.update(1)
            pbar.set_postfix({
                "COF": '{OF1:.3f}'.format(OF1=score['OF1']),
                "BOF": '{OF1:.3f}'.format(OF1=self.state['best_score']['OF1']), 
                "CCF": '{CF1:.3f}'.format(CF1=score['CF1']),
                "BCF": '{CF1:.3f}'.format(CF1=self.state['best_score']['CF1']), 
                "CM": '{map:.3f}'.format(map=score['map']),
                "BM": '{map:.3f}'.format(map=self.state['best_score']['map']), 
                "BE": '{}'.format(self.state['best_score']['best_epoch']),
            })

        return ['{OF1:.3f}'.format(OF1=self.state['best_score']['OF1']),'{CF1:.3f}'.format(CF1=self.state['best_score']['CF1']),'{map:.3f}'.format(map=self.state['best_score']['map'])]

    def train(self, data_loader, model, criterion, optimizer, epoch):

        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        end = time.time()

        for i, (input, target) in enumerate(data_loader):
            print("train")
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)


        self.on_end_epoch(True, model, criterion, data_loader, optimizer)
    
    @torch.no_grad()
    def validate(self, data_loader, model, criterion, epoch):
        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            output = self.on_forward(False, model, criterion, data_loader)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        score = self.on_end_epoch(False, model, criterion, data_loader)

        return score

    #-------------------------Bayesian dropout sampling of model output----------
    @torch.no_grad()
    def predict(self, model, dataset, samp_num, return_feature=False, return_penultimateLayer=False):
        
        output =  []

        data_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=self.state['eval_bs'], shuffle=False,
                                                   num_workers=1, collate_fn=self.state["dataset_object"].collate_fn) #self.state['workers']

        if self.state['use_gpu']:
            model = model.cuda(self.state['device_ids'][0])

        # switch to train mode
        if samp_num > 1:
            model.train()
        elif samp_num == 1:
            model.eval()
        else:
            assert 0

        for i, (input, target) in tqdm(enumerate(data_loader), total=len(data_loader)):

            self.state['input'] = input
            self.state['target'] = target

            samp_output = []
            for samp in range(samp_num):
                if return_feature:
                    model_output = self.on_forward(training=False, model=model, criterion=None, data_loader=None, return_feature=True)
                elif return_penultimateLayer:
                    model_output = self.on_forward(training=False, model=model, criterion=None, data_loader=None, return_penultimateLayer=True)
                else:
                    model_output = self.on_forward(training=False, model=model, criterion=None, data_loader=None)

                model_output = model_output.cpu().numpy().tolist()
                samp_output.append(model_output)     
            
            samp_output = np.array(samp_output).transpose(1, 0, 2).tolist()  
            
            output.extend(samp_output)

        return output

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=False):
        Trainer.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=False):
        map = self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        if display:
            if training:
                reselt_str = 'Epoch: [{0}]\t Loss {loss:.4f}\t mAP {map:.3f} \n ' \
                'OP: {OP:.3f}\t OR: {OR:.3f}\t OF1: {OF1:.3f}\t CP: {CP:.3f}\t CR: {CR:.3f}\t CF1: {CF1:.3f}'.format(
                self.state['epoch'], loss=loss, map=100 * map, OP=100 * OP, OR=100 * OR, OF1=100 * OF1, CP=100 * CP, CR=100 * CR, CF1=100 * CF1)

            else:
                reselt_str = 'Test: \t Loss {loss:.4f}\t mAP {map:.3f} \n' \
                'OP: {OP:.3f}\t OR: {OR:.3f}\t OF1: {OF1:.3f}\t CP: {CP:.3f}\t CR: {CR:.3f}\t CF1: {CF1:.3f} \n' \
                'OP_3: {OP_3:.3f}\t OR_3: {OR_3:.3f}\t OF1_3: {OF1_3:.3f}\t CP_3: {CP_3:.3f}\t CR_3: {CR_3:.3f}\t CF1_3: {CF1_3:.3f}'.format(
                    loss=loss, map=100 * map, OP=100 * OP, OR=100 * OR, OF1=100 * OF1, CP=100 * CP, CR=100 * CR, CF1=100 * CF1,
                    OP_3=100 * OP_k, OR_3=100 * OR_k, OF1_3=100 * OF1_k, CP_3=100 * CP_k, CR_3=100 * CR_k, CF1_3=100 * CF1_k)

            print(reselt_str)
            # self.result_file.write(reselt_str + '\n')

        result = {'map': map, 'OP': OP, 'OR': OR, 'OF1': OF1, 'CP': CP, 'CR': CR, 'CF1': CF1}

        return result

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=False):

        self.state['target_gt'] = self.state['target'].clone()

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=False):

        # record loss
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data.cpu(), self.state['target_gt'].cpu())

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):

        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)

        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'],
                                             'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']['OF1']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best


    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for opt in optimizer.values():
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] * decay
                lr_list.append(param_group['lr'])
        return np.unique(lr_list)

