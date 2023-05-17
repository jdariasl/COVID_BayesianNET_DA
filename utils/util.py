import torch
import os
import subprocess
import shutil
import time
from collections import OrderedDict
import json
import pandas as pd
from model.models import BDenseNet, DenseNet, BEfficientNet, EfficientNet, DA_model
import csv
import numpy as np


def write_score(writer, iter, mode, metrics):
    writer.add_scalar(mode + '/loss', metrics.data['loss'], iter)
    writer.add_scalar(mode + '/acc', metrics.data['correct'] / metrics.data['total'], iter)

def write_train_val_score(writer, epoch, train_stats, val_stats):
    writer.add_scalars('Loss', {'train': train_stats[0],
                                'val': val_stats[0],
                                }, epoch)
    writer.add_scalars('Coeff', {'train': train_stats[1],
                                 'val': val_stats[1],
                                 }, epoch)

    writer.add_scalars('Air', {'train': train_stats[2],
                               'val': val_stats[2],
                               }, epoch)

    writer.add_scalars('CSF', {'train': train_stats[3],
                               'val': val_stats[3],
                               }, epoch)
    writer.add_scalars('GM', {'train': train_stats[4],
                              'val': val_stats[4],
                              }, epoch)
    writer.add_scalars('WM', {'train': train_stats[5],
                              'val': val_stats[5],
                              }, epoch)
    return

def showgradients(model):
    for param in model.parameters():
        print(type(param.data), param.size())
        print("GRADS= \n", param.grad)

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

def save_checkpoint(state, is_best, path,  filename='last'):

    name = os.path.join(path, filename+'_checkpoint.pth.tar')
    print(name)
    torch.save(state, name)

def save_model(model, args, metrics, epoch, best_pred_loss,confusion_matrix):
    loss = metrics.data['bacc']
    save_path = args.save
    make_dirs(save_path)
    
    with open(save_path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    is_best = False
    if loss > best_pred_loss:
        is_best = True
        best_pred_loss = loss
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'metrics': metrics.data },
                        is_best, save_path, args.model + "_best")
        np.save(save_path + 'best_confusion_matrix.npy',confusion_matrix.cpu().numpy())
            
    else:
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'metrics': metrics.data},
                        False, save_path, args.model + "_last")

    return best_pred_loss

def load_model(args,weights):
    checkpoint = torch.load(args.saved_model)
    model, bflag = select_model(args,weights)
    model.load_state_dict(checkpoint['state_dict'])

    epoch = checkpoint['epoch']
    return model, epoch, bflag

def make_dirs(path):
    if not os.path.exists(path):

        os.makedirs(path)

def create_stats_files(path):
    train_f = open(os.path.join(path, 'train.csv'), 'w')
    val_f = open(os.path.join(path, 'val.csv'), 'w')
    return train_f, val_f

def read_json_file(fname):
    with open(fname, 'r') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json_file(content, fname):
    with open(fname, 'w') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def read_filepaths(file):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            if ('/ c o' in line):
                break
            #print(line)    
            subjid, path, label = line.split(' ')

            paths.append(path)
            labels.append(label)
    labes_array = np.array(labels)
    classes = np.unique(labes_array)
    for i in classes:
        print('Clase={}-Samples={}'.format(i, np.sum(labes_array == i)))
    return paths, labels

def select_model(args,weights):
    if args.model == 'BDenseNet':
        if args.init_from:
            model = BDenseNet(n_classes = args.classes, saved_model = args.saved_model)
            return DA_model(model,
                            args,
                            weights,
                            b_flag = True), True
        else:
            return DA_model(BDenseNet(n_classes = args.classes),
                            args,
                            weights,
                            b_flag=True), True #Flag: True: Bayesian model, False: Frequentist model
    elif args.model == 'DenseNet':
        return DA_model(DenseNet(n_classes = args.classes),
                        args,
                        weights), False
    elif args.model == 'EfficientNet':
        return DA_model(EfficientNet(n_classes = args.classes),
                        args,
                        weights), False
    elif args.model == 'BEfficientNet':
        if args.init_from:
            model = BEfficientNet(n_classes = args.classes, saved_model = args.saved_model)
            return DA_model(model,
                            args,
                            weights,
                            b_flag=True), True
        else:
            return DA_model(BEfficientNet(n_classes = args.classes),
                            args,
                            weights,
                            b_flag=True), True

def ImportantOfContext(ReMap: np.array, Mask: np.array) -> float:
    (rr,cr) = ReMap.shape
    (rm,cm) = Mask.shape
    assert rr == rm, 'Relevance Map and Mask mismatch in the number of rows'
    assert cr == cm, 'Relevance Map and Mask mismatch in the number of columns'
    
    Mask[Mask>0] = 1
    ReMap[ReMap<0]=0 #Take only pixels with positive relevance to estimate IoC
    
    Pin = ReMap * Mask
    npin = np.sum(Pin > 0)
    
    Pout = ReMap * (1 - Mask)
    npout = np.sum(Pout > 0)
    
    IoC = (np.sum(Pout)/npout)/(np.sum(Pin)/npin)
    return IoC

def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2, wait=False, sleep_time=10):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere

    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696

    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]  # Remove garbage
        # Keep gpus under threshold only
        free_gpus = [
            str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
        ]
        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join(free_gpus)
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
    print(f"Using GPU(s): {gpus_to_use}")
    #logger.info(f"Using GPU(s): {gpus_to_use}")