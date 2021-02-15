import torch
import mlflow
import torch.nn.functional as F

import time

#from torch.autograd import Variable


def log_scalar(name, value, step, writer):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)


# def train(use_cuda, model, epoch, optimizer, log_interval, train_loader, writer):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         if use_cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
#                   f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
#             step = epoch * len(train_loader) + batch_idx
#             log_scalar('train_loss', loss.data.item(), step, writer)
#             model.log_weights(step, writer)


# def test(use_cuda, model, epoch, test_loader, writer):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             if use_cuda:
#                 data, target = data.cuda(), target.cuda()
#             data, target = Variable(data), Variable(target)
#             output = model(data)
#             # sum up batch loss
#             test_loss += F.nll_loss(output, target,
#                                     reduction='sum').data.item()
#             # get the index of the max log-probability
#             pred = output.data.max(1)[1]
#             correct += pred.eq(target.data).cpu().sum().item()

#     test_loss /= len(test_loader.dataset)
#     test_accuracy = 100.0 * correct / len(test_loader.dataset)
#     print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.0f}%)\n')
#     step = (epoch + 1) * len(test_loader)
#     log_scalar('test_loss', test_loss, step, writer)
#     log_scalar('test_accuracy', test_accuracy, step, writer)

##################################

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CyclicLR, StepLR

from utils import split_train_val, count_model_parameters
from metrics import iou_fnc, accuracy, compute_per_channel_dice
from dataset_objects import VolumeDataset
from losses import FocalLoss

########

def eval_training(net, test_loader, gpu=False, n_class=2, weights=None, device=None):
    """Evaluation during trainig"""
    
    tot = 0
    acc = 0
    iou = 0

    iou_sum = np.zeros(n_class)
    count_sum = np.zeros(n_class)

    i = 0

    if weights == None:
        w_vec = torch.ones(n_class)
    else:
        w_vec = weights

    #e_criterion = nn.CrossEntropyLoss(weight=w_vec)
    e_criterion = FocalLoss(apply_nonlin=None, alpha=0.5, gamma=2, smooth=1e-5)

    
    if gpu:
        e_criterion.cuda()
        e_criterion.to(device)

    net.eval()
    with torch.no_grad():
    
        for data in test_loader:

            imgs, true_masks = data
            
            if gpu:                                
                imgs = imgs.to(device, non_blocking=False) #, non_blocking=True) #faster
                true_masks = true_masks.to(device, non_blocking=False)

            masks_pred = net(imgs)
            #masks_probs = F.softmax(masks_pred, dim=1)

            ##tot += e_criterion(masks_pred, true_masks)
            #tot += e_criterion(masks_pred, true_masks.type(torch.long))

            ##for focal loss
            tot += e_criterion(F.softmax(masks_pred, dim=1), true_masks.type(torch.long))

            #metrics
            out = torch.argmax(masks_pred, dim=1).float()
            
            acc += accuracy(out, true_masks)
                        
            iter_iou, iter_count = iou_fnc(out, true_masks, n_class)
            
            iou_sum = iou_sum + iter_iou
            count_sum = count_sum + iter_count

            i +=1

    ##IoU
    iou_scores = iou_sum / (count_sum + 1e-10)
    iou_mean = np.nanmean(iou_scores)

    for c in range(n_class):
        if count_sum[c] == 0.0:
            iou_scores[c] = np.nan #float('nan')
        print('class {} IoU: {}'.format(c, iou_scores[c]))

    ##########################


    net.train()

    return tot / i, acc / i, iou_mean

##########################################

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.01,
              lr_step_size=100,
              lr_gamma=0.1,
              test_percent=0.15,
              save_cp=True,
              gpu=False,
              class_weights=None,
              test_epochs=10,
              dataset_size=10,
              n_class=2,
              dataset_path='',
              checkpoint_path='',
              device=None,
              writer=None):

    ids = list(range(dataset_size))
    iddataset = split_train_val(ids, test_percent)

    # iddataset['train'] = [1, 2, 5, 8]
    # iddataset['val'] = [3, 9]

    iddataset['train'] = [48, 3, 23, 75, 22, 109, 73, 130, 115, 86, 121, 67, 97, 116, 14, 125, 52, 84, 129, 58, 49, 110,
                        43, 88, 25, 4, 89, 50, 29, 94, 53, 16, 2, 46, 92, 113, 44, 15, 111, 124, 69, 47, 5, 104, 54, 37, 76,
                        119, 13, 34, 21, 103, 80, 91, 82, 35, 19, 6, 72, 59, 105, 83, 20, 128, 120, 57, 101, 30, 28, 24, 8, 41,
                        31, 95, 63, 0, 126, 11, 1, 85, 7, 33, 127, 56, 118, 70, 26, 81, 78, 40, 55, 122, 99, 71, 60, 42, 87, 9, 93,
                        108, 39, 18, 77, 90, 68, 32, 102, 79, 12, 96, 112, 36]
    iddataset['val'] = [17, 64, 27, 114, 74, 45, 61, 38, 106, 100, 117, 51, 62, 65, 123, 66, 10, 107, 98]

    print("train ids:" + str(iddataset['train']))
    print("val ids:" + str(iddataset['val']))
    print("************************************")

    ###################################
    ## for local testing
    # ids = list(range(20))
    # iddataset = split_train_val(ids, 0.5)
    # iddataset['train'] = iddataset['train'][:3]
    # iddataset['val'] = iddataset['val'][:2]


    if class_weights == None:
        class_weights = np.ones(n_class)
    else:
        class_weights = np.array([float(i) for i in class_weights.split(',')])

    ###################################################

    ## apply_trans=False to disable random data perturbation, also shuffle=False
    train_set = VolumeDataset(iddataset['train'], path=dataset_path, apply_trans=False, n_class=n_class)
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              shuffle=False, #False for rep!!!!!!!!
                                              num_workers=2, ##########
                                              pin_memory=False,
                                              sampler=None)

    test_set = VolumeDataset(iddataset['val'], path=dataset_path, apply_trans=False, n_class=n_class)
    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=False,
                                            num_workers=1) # pin_memory=True??

    ###########################################################

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Class weights: {}
        Training size: {}
        Validation size: {}
        Input shape: {}
        Checkpoints: {}
        CUDA: {}
        Model params: {}
    '''.format(epochs,
               batch_size,
               lr,
               class_weights,
               len(iddataset['train']),
               len(iddataset['val']),
               train_set[0][0].shape,
               str(save_cp),
               str(gpu),
               str(count_model_parameters(net)) ))


    ##############################################
    # Optimizer
    
    # optimizer = optim.SGD(net.parameters(),
    #                       lr=lr,
    #                       momentum=0.9, #)
    #                       weight_decay=0.0001)

    optimizer = optim.Adam(params=net.parameters(), lr=lr,
                            weight_decay=0.0001) #weight_decay=0.0001

    
    #########################
    # LR scheduler

    #lr adjust
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma) #step_size=800, gamma=0.1) #every 800 epoch, one order of mag. red.
    #########################
    # Criterion

    w_vec = torch.tensor(class_weights, dtype=torch.float)
    #criterion = nn.CrossEntropyLoss(weight=w_vec)

    criterion = FocalLoss(apply_nonlin=None, alpha=0.5, gamma=2, smooth=1e-5)

    ############################

    if gpu:
      #criterion.cuda()
      criterion.to(device)

    start_time = time.time()

    net.train()

    print('Starting epochs ...')
    for epoch in range(epochs):

        iou_sum = np.zeros(n_class)
        count_sum = np.zeros(n_class)

        #print('LR: ', scheduler.get_lr())
        print('lr: ', scheduler.get_last_lr()[0])

        epoch_loss = 0
        acc = 0.0

        
        i = 0
        for data in train_loader:
            
            imgs, true_masks = data

            if gpu:

                imgs = imgs.to(device, non_blocking=False) #, non_blocking=True) # faster, but can cause sync errors
                true_masks = true_masks.to(device, non_blocking=False)
 
            masks_probs = net(imgs)
            
            #loss = criterion(masks_probs, true_masks.type(torch.long))

            ##for focal loss
            loss = criterion(F.softmax(masks_probs, dim=1), true_masks.type(torch.long))
            
            #########################################################################################
            #eval metric

            acc += accuracy(torch.argmax(masks_probs, dim=1).float(), true_masks)

            iter_iou, iter_count = iou_fnc(torch.argmax(masks_probs, dim=1).float(), true_masks, n_class)
            iou_sum = iou_sum + iter_iou
            count_sum = count_sum + iter_count
            
            #########################################################################################
            
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            i +=1

        ###adjust lr per epoch
        scheduler.step()

        #timestamp
        ts = ((time.time() - start_time)/(60*60))

        ##################
        #iou per epoch
        iou_scores = iou_sum / (count_sum + 1e-10)
        iou_mean = np.nanmean(iou_scores)

        print('epoch {0:.1f} - loss: {1:.15f} - acc: {2:.15f} - meanIoU: {3:.15f} - ts: {4:.15f}'.format(epoch + 1, epoch_loss/i, acc/i, iou_mean, ts))

        #########################################
        
        for c in range(n_class):
            if count_sum[c] == 0.0:
                iou_scores[c] = np.nan #float('nan')
            print('class {} IoU: {}'.format(c, iou_scores[c]))

        ################################
        #log
        log_scalar('train_loss', epoch_loss/i, epoch + 1, writer)
        log_scalar('train_iou', iou_mean, epoch + 1, writer)
        ################################
        
        
        if epoch%test_epochs == 0:
        #if epoch%2 == 0: #for LR range test

            print('eval ' + str(epoch +1) + ' ..................................................')
            val_loss, acc, m_iou = eval_training(net, test_loader, gpu=gpu, n_class=n_class, weights=w_vec, device=device)
            
            #print('Eval_acc: {}'.format(acc))
            print('eLoss: {0:.15f} - eAcc: {1:.15f} - eMeanIoU: {2:.15f}'.format(val_loss, acc, m_iou))

            ################################
            #log
            log_scalar('test_loss', float(val_loss), epoch + 1, writer)
            log_scalar('test_iou', float(m_iou), epoch + 1, writer)
            ################################

        if save_cp and epoch%10 == 0:
            
            mlflow.pytorch.log_model(net, 'model_CP_' + str(epoch + 1))
            #net.log_weights(epoch + 1, writer) ???

            #torch.save(net.state_dict(),
            #           checkpoint_path + 'CP_{}.pth'.format(epoch + 1))
            #print('Checkpoint {} saved !'.format(epoch + 1))
