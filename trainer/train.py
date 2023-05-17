import torch
import os
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
from model.loss import crossentropy_loss
from utils.util import select_model, load_model, assign_free_gpus
from utils.util_model import Metrics, print_summary
from model.metric import accuracy
from COVIDXDataset.dataset import COVIDxDataset
from torch.utils.data import DataLoader


def initialize(args):
    
    train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=args.dataset,
                                 dim=(224, 224), pre_processing = args.pre_processing)
    #print(train_loader.)
    #------ Class weigths for sampling and for loss function -----------------------------------
    labels_class = np.unique(train_loader.labels['label_class'])
    print(labels_class)
    class_weight = compute_class_weight(class_weight='balanced', classes=labels_class, y=train_loader.labels['label_class'])
    
    labels_db = np.unique(train_loader.labels['label_db'])
    db_weight = compute_class_weight(class_weight='balanced', classes=labels_db, y=train_loader.labels['label_db'])
    #---------- Alphabetical order in labels does not correspond to class order in COVIDxDataset-----
    class_weight = class_weight[::-1]

    if args.device is not None:
        assign_free_gpus()
    model, bflag = select_model(args,[class_weight,db_weight])
    
    #-------------------------------------------
    val_loader = COVIDxDataset(mode='test', n_classes=args.classes, dataset_path=args.dataset,
                               dim=(224, 224), pre_processing = args.pre_processing)
    #------------------------------------------------------------------------------------
    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 4}#'sampler' : sampler
    
    test_params = {'batch_size': args.batch_size,
                   'shuffle': True,
                   'num_workers': 4}
    #------------------------------------------------------------------------------------------
    training_generator = DataLoader(train_loader, **train_params)
    val_generator = DataLoader(val_loader, **test_params)
    return model,training_generator,val_generator, bflag

def validation(args, model, testloader, epoch):
    model.eval()

    #-------------------------------------------------------
    #Esto es para congelar las capas de la red preentrenada
    #for m in model.modules():
    #    if isinstance(m, nn.BatchNorm2d):
    #        m.train()
    #        m.weight.requires_grad = False
    #        m.bias.requires_grad = False
    #-----------------------------------------------------

    metrics = Metrics('')
    metrics.reset()
    confusion_matrix = torch.zeros(args.classes, args.classes)
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):

            input_data, target = input_tensors
            
            #print(input_data.shape)
            output = model(input_data)
            
            loss = crossentropy_loss(output, target['label_class'],weight=model.class_weight)

            correct, total, acc = accuracy(output, target)
            #num_samples = batch_idx * args.batch_size + 1
            _, preds = torch.max(output, 1)
            bacc = balanced_accuracy_score(target.cpu().detach().numpy(),preds.cpu().detach().numpy())
            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})
            #print_stats(args, epoch, num_samples, testloader, metrics)

    print_summary(args, epoch, batch_idx, metrics, mode="Validation")
    return metrics,confusion_matrix

def validation_bayesian(args, model, testloader, epoch):
    model.eval()

    metrics = Metrics('')
    metrics.reset()
    confusion_matrix = torch.zeros(args.classes, args.classes)
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):

            input_data, target = input_tensors
           
            output = model(input_data)
            loss = crossentropy_loss(output, target['label_class'],weight=model.class_weight)

            with torch.no_grad():
                output_mc = []
                for _ in range(args.n_monte_carlo):
                    logits = model(input_data)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    output_mc.append(probs)
                output = torch.stack(output_mc)  
                pred_mean = output.mean(dim=0)

            correct, total, acc = accuracy(pred_mean, target)
            num_samples = batch_idx * args.batch_size + 1
            _, preds = torch.max(pred_mean, 1)
            bacc = balanced_accuracy_score(target.cpu().detach().numpy(),preds.cpu().detach().numpy())
            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})
            #print_stats(args, epoch, num_samples, testloader, metrics)

    print_summary(args, epoch, num_samples, metrics, mode="Validation")
    return metrics,confusion_matrix
    
def initialize_from_saved_model(args):
    print('Training on saved model')

    train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=args.dataset,
                                 dim=(224, 224), pre_processing = args.pre_processing)
    #print(train_loader.)
    #------ Class weigths for sampling and for loss function -----------------------------------
    labels_class = np.unique(train_loader.labels['label_class'])
    print(labels_class)
    class_weight = compute_class_weight(class_weight='balanced', classes=labels_class, y=train_loader.labels['label_class'])
    
    labels_db = np.unique(train_loader.labels['label_db'])
    db_weight = compute_class_weight(class_weight='balanced', classes=labels_db, y=train_loader.labels['label_db'])
    #---------- Alphabetical order in labels does not correspond to class order in COVIDxDataset-----
    class_weight = class_weight[::-1]
    #-------------------------------------------
    if args.device is not None:
        assign_free_gpus()
    model, epoch, bflag = load_model(args,[class_weight,db_weight])
    #-------------------------------------------
    val_loader = COVIDxDataset(mode='test', n_classes=args.classes, dataset_path=args.dataset,
                            dim=(224, 224), pre_processing = args.pre_processing)
    #------------------------------------------------------------------------------------
    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 4}#'sampler' : sampler
    test_params = {'batch_size': args.batch_size,
                    'shuffle': False,
                    'num_workers': 4}
    #------------------------------------------------------------------------------------------
    training_generator = DataLoader(train_loader, **train_params)
    val_generator = DataLoader(val_loader, **test_params)
    return model, training_generator,val_generator, epoch, bflag