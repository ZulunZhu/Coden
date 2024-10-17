import time
import uuid
import random
import argparse
import gc
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from utils import SimpleDataset
from model import ClassMLP
from utils import *
from glob import glob
from tqdm import tqdm

from sklearn import preprocessing as sk_prep
from sklearn import metrics
import logging
import copy
import os
import sys
from sys import getsizeof
import update_grad
import pynvml
import resource
from sklearn.metrics import average_precision_score, roc_auc_score
from openTSNE import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from varname import nameof
from texttable import Texttable
def update_results(model_name, snap_number, data, file_path='model_results.csv'):
    # Ensure the snap_number is formatted correctly, e.g., "snap_1"
    snap_column = f'snap_{snap_number}'

    if os.path.exists(file_path):
        # Load the existing CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col=0)  # Assuming the first column is the index (model names)
    else:
        # Create a new DataFrame if the file does not exist
        df = pd.DataFrame()

    # Update or create the model entry
    if model_name in df.index:
        # If the model already exists in the DataFrame
        df.at[model_name, snap_column] = data
    else:
        # If the model does not exist, add it to the DataFrame
        df.loc[model_name, snap_column] = data

    # Save the updated DataFrame to CSV
    df.to_csv(file_path)
def show_gpu():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("GPU overall::",meminfo.total/1024**3, "GB") #总的显存大小
    print("GPU allocated::",meminfo.used/1024**3, "GB")  #已用显存大小
    print("GPU left::",meminfo.free/1024**3, "GB")  #剩余显存大小



def update_results_csv(result_path, model_name, dataset_name, new_result):
    # Load the existing CSV file into a DataFrame or create a new one if it doesn't exist
    try:
        df = pd.read_csv(result_path, index_col=0)
    except FileNotFoundError:
        # If the file does not exist, create an empty DataFrame with model_name as the index
        df = pd.DataFrame(columns=[dataset_name])
        df.index.name = 'Model'

    # Check if the dataset_name column exists, if not, add it
    if dataset_name not in df.columns:
        df[dataset_name] = pd.NA  # Initialize the column with NA values

    # Ensure the model_name row exists
    if model_name not in df.index:
        # If the model doesn't exist, append a new row with NA values
        new_row = pd.DataFrame(index=[model_name], columns=df.columns)
        df = pd.concat([df, new_row])

    # Update the specific entry with the new result
    df.at[model_name, dataset_name] = new_result

    # Save the updated DataFrame back to CSV
    df.to_csv(result_path)

    print(f"Updated CSV file at {result_path} with new results for model '{model_name}' and dataset '{dataset_name}'.")

def tsne_plt(embeddings, labels, save_path=None, title='Title'):
    print('Drawing t-SNE plot ...')
    tsne = TSNE(n_components=3, perplexity=30, metric="euclidean", n_jobs=8, random_state=42, verbose=False)
    embeddings = embeddings.cpu().numpy()
    c = labels.cpu().numpy()

    emb = tsne.fit(embeddings)  # Training

    plt.figure(figsize=(10, 8))
    plt.scatter(emb[:, 0], emb[:, 1], c=c, marker='o')
    plt.colorbar()
    plt.grid(True)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def tab_printer(args):
    """Function to print the logs in a nice tabular format.
    
    Note
    ----
    Package `Texttable` is required.
    Run `pip install Texttable` if was not installed.
    
    Parameters
    ----------
    args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," "), args[k]] for k in keys])
    print(t.draw())
def main():
    parser = argparse.ArgumentParser()
    
    
    # Dataset and Algorithom
    parser.add_argument('--seed', type=int, default=20159, help='random seed..')
    parser.add_argument('--alg', default='instant', help='push algorithm')
    parser.add_argument('--cl_alg', default='PGL', help='contrastive learning algorithm')
    parser.add_argument('--dataset', default='papers100M', help='dateset.')
    # Algorithm parameters
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha.')
    parser.add_argument('--rmax', type=float, default=1e-7, help='threshold.')
    parser.add_argument('--rbmax', type=float, default=1, help='reverse push threshold.')
    parser.add_argument('--delta', type=float, default=0.1, help='sample threshold.')

    parser.add_argument('--epsilon', type=float, default=8, help='epsilon.')
    parser.add_argument("--n-ggd-epochs", type=int, default=1,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--drop_feat", type=float, default=0.1,
                        help="feature dropout rate")
    parser.add_argument('--use_gcl', default="yes", help='bias.')
    # Learining parameters
    
    parser.add_argument("--classifier-lr", type=float, default=0.05, help="classifier learning rate")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')
    parser.add_argument('--layer', type=int, default=4, help='number of layers.')
    parser.add_argument('--hidden', type=int, default=2048, help='hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate.')
    parser.add_argument('--bias', default='none', help='bias.')
    parser.add_argument("--proj_layers", type=int, default=1, help="number of project linear layers")
    parser.add_argument('--epochs', type=int, default= 100, help='number of epochs.')
    parser.add_argument('--batch', type=int, default=2048, help='batch size.')
    parser.add_argument("--patience", type=int, default=100, help="early stop patience condition")
    parser.add_argument('--dev', type=int, default=0, help='device id.')
    parser.add_argument('--skip_sn0', type=int, default=1, help='decide whether to skip snapshot 0.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    tab_printer(args)

    free_gpu_id = int(get_free_gpu())
    torch.cuda.set_device(free_gpu_id)
    # torch.cuda.set_device(args.dev)
    checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

    
    n,m,features,features_n,train_labels,val_labels,test_labels,train_idx,val_idx,test_idx,memory_dataset, py_alg = load_ogb_init(args.dataset, args.alpha,args.rmax, args.rbmax,args.delta,args.epsilon, args.alg) ##
    print("features_before_training:",features)
    print("train_labels",torch.sum(train_labels))
    
    print('------------------ Initial -------------------')
    print("train_idx:",train_idx.size())
    print("val_idx:",val_idx.size())
    print("test_idx:",test_idx.size())
    macros = []
    micros = [] 
    x_max = np.max(np.abs(features_n), axis=1)
    x_norm = np.sum(x_max)
    print("x_max", x_max)
    print("x_norm", x_norm)
    x_max=np.array(x_max,dtype=np.float64)
    x_max = np.ascontiguousarray(x_max)

    pretrain_times = []
    # change_node_list = np.zeros([n]) 
    # snapList = [f for f in glob('../data/'+args.dataset+'/*_feat_*.npy')]
    # feature_list = []
    # print(len(snapList))
    # for i in range(len(snapList)):
    #     features = np.load('../data/'+args.dataset+'/'+args.dataset+'_feat_'+str(i)+'.npy')
    #     assert features.shape[1]==128
    #     features = torch.FloatTensor(features)
    #     feature_list.append(features)
    # features = torch.stack(feature_list, dim=1)
    # # features = feature_list[-1]
    # # features = features.unsqueeze(1)
    # print(features.shape)
    

    if not args.skip_sn0:
        macro_init, micro_init= prepare_to_train(0,features, m, train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, args, checkpt_file, change_node_list)
        macros.append(macro_init)
        micros.append(micro_init)
    arxiv_table_path = Path('../mag_accuracy_table.csv')

    print('------------------ update -------------------')
    snapList = [f for f in glob('../data/'+args.dataset+'/*Edgeupdate_snap*.txt')]
    print('number of snapshots: ', len(snapList))
    # print("features[2][3]::",features[2][3])
    for i in range(len(snapList)):
        change_node_list = np.zeros([1])
        py_alg.split_batch('../data/'+args.dataset+'/'+args.dataset+'_Edgeupdate_snap'+str(i+1)+'.txt', args.delta, x_max, x_norm)
        split_List = [f for f in glob('../data/'+args.dataset+'/*Edgeupdate_snap'+str(i+1)+'/edges_part_*.txt')]
        print("split_List",len(split_List))
        feature_list = []
        
        for j in range(len(split_List)):
            py_alg.snapshot_lazy('../data/'+args.dataset+'/'+args.dataset+'_Edgeupdate_snap'+str(i+1)+'/edges_part_'+str(j)+'.txt', args.rmax, args.rbmax,args.delta, args.alpha, features, change_node_list, args.alg)
            # py_alg.snapshot_lazy('../data/'+args.dataset+'/'+args.dataset+'_Edgeupdate_snap'+str(i+1)+'.txt', args.rmax, args.rbmax,args.delta, args.alpha, features, change_node_list, args.alg)
            feature_list.append(features)
        
        # features = feature_list[-1]
        # features = torch.from_numpy(features).unsqueeze(0).float()
        
        # features = features.permute(1, 0, 2)
        # Convert the feature_list to a PyTorch tensor
        feature_step = torch.FloatTensor(np.array(feature_list))
        print(feature_step.shape)
        feature_step = feature_step.permute(1, 0, 2)
        
        # print("features", features)
        # print("feature_step", feature_step)
        # exit(0)
        # if i<15:
        #     continue
        macro, micro = prepare_to_train(i+1,feature_step, m,train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, args, checkpt_file, change_node_list)
        macros.append(macro)
        micros.append(micro)

   
        # # update_results_csv(arxiv_table_path, "negative samples", str(args.delta), features_n.shape[0])
        # update_results_csv(arxiv_table_path, "accuracy", i, micro)
        

        # with open('./log/sensitivity.txt', 'a') as f:
        #     print('Dataset:'+args.dataset+f"metric_micro:{100*np.mean( micros):.2f}%  "+f" hidden: {args.hidden:.1f}"+f" epochs: {args.n_ggd_epochs:.1f}",file=f)
        # exit(0)
        
        
    print("Mean Macro: ", np.mean( macros), " Mean Micro: ", np.mean( micros), "Mean training time: ", np.mean(pretrain_times))
    # with open('./log/sensitivity.txt', 'a') as f:
    #     print('Dataset:'+args.dataset+f"metric_micro:{100*np.mean( micros):.2f}%  "+f" hidden: {args.hidden:.1f}"+f" epochs: {args.n_ggd_epochs:.1f}",file=f)

def train_incremental(model, device, train_loader, optimizer, mem_loader):
    model.train()

    time_epoch=0
    loss_list=[]
    loss_fun = nn.BCEWithLogitsLoss()
    for step, (x, y) in enumerate(train_loader):
        t_st=time.time()
        task_grads = {}
        for step_mem, (x_mem, y_mem) in enumerate(mem_loader):
            x_mem, y_mem = x_mem.cuda(), y_mem.cuda()
            optimizer.zero_grad()
            out_mem = model(x_mem)
            loss_mem = F.nll_loss(out_mem, y_mem.squeeze(1))
            # print("out", out.sigmoid())
            # print("y.squeeze(1)", y.to(torch.float))
            # if(torch.sum(y)>0):
            #     print("torch.sum(y)>0!!")
            # loss = loss_fun(out.sigmoid(), y.to(torch.float))
            loss_mem.backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                gradients[name] = parameter.grad.clone()
            task_grads[step_mem] = update_grad.grad_to_vector(gradients)
        ref_grad_vec = torch.stack(list(task_grads.values()))
        ref_grad_vec = torch.sum(ref_grad_vec, dim=0)/ref_grad_vec.shape[0]

        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        # print("out", out.sigmoid())
        # print("y.squeeze(1)", y.to(torch.float))
        # if(torch.sum(y)>0):
        #     print("torch.sum(y)>0!!")
        # loss = loss_fun(out.sigmoid(), y.to(torch.float))
        loss.backward()
        # for name, parameter in model.named_parameters():
        #     print(parameter.grad)
        # Example: save gradients as a torch file
        gradients = {}
        for name, parameter in model.named_parameters():
            gradients[name] = parameter.grad.clone()  # Use `.clone()` to save a copy of the gradient tensor
        # torch.save(gradients, grad_checkpt_file)
        # loaded_gradients = torch.load(grad_checkpt_file)

    #    for param in classifier.parameters():
    #     print(param)
    #    print("loaded_gradients", loaded_gradients)
        # for n, p in gradients.items():
        #     print("loaded_gradients", p)
        current_grad_vec = update_grad.grad_to_vector(gradients)
        # print("current_grad_vec", current_grad_vec)
        # print("ref_grad_vec", ref_grad_vec)

        assert current_grad_vec.shape == ref_grad_vec.shape
        dotp = current_grad_vec * ref_grad_vec
        dotp = dotp.sum()
        if (dotp < 0).sum() != 0:
            # new_grad = update_grad.get_grad(current_grad_vec, ref_grad_vec)
            new_grad = update_grad.get_grad(current_grad_vec,ref_grad_vec)
            # copy gradients back
            # print("current_grad_vec", current_grad_vec)
            # print("new_grad", new_grad)
            update_grad.vector_to_grad(model,new_grad)
            # for name, parameter in model.named_parameters():
            #     print(parameter.grad)
            # exit(0)
        
        optimizer.step()
        time_epoch+=(time.time()-t_st)
        loss_list.append(loss.item())
        
    return np.mean(loss_list), time_epoch

def train(model, device, train_loader, optimizer):
    model.train()

    time_epoch=0
    loss_list=[]
    loss_fun = nn.BCEWithLogitsLoss()
    for step, (x, y) in enumerate(train_loader):
        t_st=time.time()
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        # print("out", out.sigmoid())
        # print("y.squeeze(1)", y.to(torch.float))
        # if(torch.sum(y)>0):
        #     print("torch.sum(y)>0!!")
        # loss = loss_fun(out.sigmoid(), y.to(torch.float))
        loss.backward()
        optimizer.step()
        time_epoch+=(time.time()-t_st)
        loss_list.append(loss.item())
        
    return np.mean(loss_list), time_epoch

def custom_evaluator(y_true, y_pred):
    # Convert tensors to numpy arrays
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # Calculate accuracy
    accuracy = accuracy_score(y_true_np, y_pred_np)
    return accuracy

@torch.no_grad()
def validate(model, device, loader):
    model.eval()
    y_pred, y_true = [], []
    for step,(x, y) in enumerate(loader):
        x = x.cuda()
        out = model(x)
        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_true.append(y)
    # return evaluator.eval({
    #     "y_true": torch.cat(y_true, dim=0),
    #     "y_pred": torch.cat(y_pred, dim=0),
    # })['acc']
    return custom_evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))


@torch.no_grad()
def test(model, device, loader,checkpt_file):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    y_pred, y_true = [], []
    for step,(x, y) in enumerate(loader):
        x = x.cuda()
        out = model(x)
        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_true.append(y)
    metric_macro = metrics.f1_score(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0), average='macro')
    metric_micro = metrics.f1_score(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0), average='micro')
    # For mooc and reddit datasets
    # roc = roc_auc_score(torch.cat(y_true, dim=0),torch.cat(y_pred, dim=0))
    roc = 0
    return custom_evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)), metric_macro, metric_micro, roc

def aug_feature_dropout(input_feat, drop_percent=0.2):
    # aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def prepare_to_train(snapshot, features_ori, m, train_idx, val_idx, test_idx, train_labels, val_labels, test_labels, args, checkpt_file, change_node_list):
    features_ori = torch.FloatTensor(features_ori)
    
    n = features_ori.size()[0]
    feature_dim = features_ori.size(-1)
    print("Original feature size: ", features_ori.size(0))
    print("train_idx:", train_idx.size())
    print("n=", n, " m=", m)
    all_labels = torch.zeros(features_ori.size(0),dtype=torch.int64)

    label_dim = int(max(train_labels.max(),val_labels.max(),test_labels.max()))+1
    labels = torch.cat((train_labels, val_labels,test_labels)).squeeze(1).cuda()
    
    print("labels:",labels.size())
    
    
    print("features.size:", features_ori.shape)
 
    print("labels.size(0):", labels.size(0))


    print("GPU used::",torch.cuda.memory_allocated()/1024/1024,"MB")

    show_gpu()
    
    classifier = ClassMLP(snapshot,feature_dim,args.hidden,label_dim,args.layer,args.dropout).cuda()
    ### Instant original method
    # features_ori = features_ori.unsqueeze(1)
    train_dataset = ExtendDataset(features_ori[train_idx],train_labels)
    valid_dataset = ExtendDataset(features_ori[val_idx],val_labels)
    test_dataset = ExtendDataset(features_ori[test_idx],test_labels)

    # all_loader = DataLoader(all_dataset, batch_size=args.batch,shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch,shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    # if snapshot>0:
        
    #     sub_train = np.where(change_node_list)[0]
    #     print("sub_train:", sub_train.shape)
    #     mini_train_dataset = ExtendDataset(embeds[sub_train],labels[sub_train].unsqueeze(1))
    #     mini_train_loader = DataLoader(mini_train_dataset, batch_size=args.batch,shuffle=True)

    #     memory = np.where(~change_node_list)[0]
    #     # Ensure that there are enough samples in memory to match sub_train
    #     if len(memory) >= len(sub_train):
    #         # Sample indices from memory
    #         memory = np.random.choice(memory, size=len(sub_train), replace=False)
    #         print("Sampled indices from memory:", memory.shape)
    #     else:
    #         print("Not enough elements in memory to match the size of sub_train.")

    #     mem_dataset = ExtendDataset(embeds[memory],labels[memory].unsqueeze(1))
    #     mem_loader = DataLoader(mem_dataset, batch_size=args.batch,shuffle=True)
    # print(classifier)
    

    classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    
    bad_counter = 0
    best = 0
    best_epoch = 0
    train_time = 0
    checkpt_file = 'pretrained/'+'snapshot'+str(snapshot)+'.pt'
    grad_checkpt_file = 'pretrained/'+'grad_snapshot'+str(snapshot)+'.pt'

    # #Initialize the model with the random parameters    
    # classifier.reset_parameters()

    #Initialize the last snapshot for initialization
    # if snapshot > 0:
    #     checkpt_file_for_initial = 'pretrained/'+'snapshot'+str(snapshot-1)+'.pt'
    #     grad_checkpt_file_for_initial = 'pretrained/'+'grad_snapshot'+str(snapshot-1)+'.pt'
    #     print("Load the last snapshot for initialization:"+checkpt_file_for_initial)
    #     classifier.load_state_dict(torch.load(checkpt_file_for_initial))
    #     loaded_gradients = torch.load(grad_checkpt_file_for_initial)
    # #    for param in classifier.parameters():
    # #     print(param)
    # #    print("loaded_gradients", loaded_gradients)
    #     # for n, p in loaded_gradients.items():
    #     #     print("loaded_gradients", p)

    #     ref_grad_vec = update_grad.grad_to_vector(loaded_gradients)
    #     torch.set_printoptions(precision=8)
    

    
    # print("current_grad_vec.shape", current_grad_vec)
    # Printing parameter values without their names
    
    
    print("--------------------------")
    print("Training...")
    if snapshot==0:
        for epoch in range(args.epochs):
            loss_tra,train_ep = train(classifier,args.dev,train_loader,classifier_optimizer)
            t_st=time.time()
            f1_val = validate(classifier, args.dev, valid_loader)
            train_time+=train_ep
            if(epoch+1)%1 == 0:
                print(f'Epoch:{epoch+1:02d},'
                f'Train_loss:{loss_tra:.3f}',
                f'Valid_acc:{100*f1_val:.2f}%',
                f'Time_cost:{train_ep:.3f}/{train_time:.3f}')
            if f1_val > best:
                best = f1_val
                best_epoch = epoch+1
                t_st=time.time()
                torch.save(classifier.state_dict(), checkpt_file)

                # Example: save gradients as a torch file
                # gradients = {}
                # for name, parameter in classifier.named_parameters():
                #     gradients[name] = parameter.grad.clone()  # Use `.clone()` to save a copy of the gradient tensor
                #     # print("parameter.grad.clone()", parameter.grad.clone())
                # torch.save(gradients, grad_checkpt_file)

                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
    
    # For incremental model test, to test whether the model of last moment can be used in the next moment (seems no)
    # if snapshot > 0:
    #    checkpt_file = 'pretrained/'+'snapshot'+str(snapshot-1)+'.pt'
    #    print("*****************************"+checkpt_file)
    if snapshot>0:
        # test_acc, metric_macro, metric_micro, roc = test(classifier, args.dev, test_loader, evaluator,checkpt_file_for_initial)
        # print(f"Train cost: {train_time:.2f}s")
        # print('Load {}th epoch'.format(best_epoch))
        # print("Checkpt_file: ", checkpt_file_for_initial)
        # print(f"Test accuracy:{100*test_acc:.2f}%")
        # print(f"metric_macro:{100*metric_macro:.2f}%")
        # print(f"metric_micro:{100*metric_micro:.2f}%")
        for epoch in range(args.epochs):
            # loss_tra,train_ep = train_incremental(classifier,args.dev,train_loader,classifier_optimizer,mini_train_loader)
            loss_tra,train_ep = train(classifier,args.dev,train_loader,classifier_optimizer)
            t_st=time.time()
            f1_val = validate(classifier, args.dev, valid_loader)
            train_time+=train_ep
            if(epoch+1)%1 == 0:
                print(f'Epoch:{epoch+1:02d},'
                f'Train_loss:{loss_tra:.3f}',
                f'Valid_acc:{100*f1_val:.2f}%',
                f'Time_cost:{train_ep:.3f}/{train_time:.3f}')
            if f1_val > best:
                best = f1_val
                best_epoch = epoch+1
                t_st=time.time()
                torch.save(classifier.state_dict(), checkpt_file)

                # # Example: save gradients as a torch file
                # gradients = {}
                # for name, parameter in classifier.named_parameters():
                #     gradients[name] = parameter.grad.clone()  # Use `.clone()` to save a copy of the gradient tensor
                #     # print("parameter.grad.clone()", parameter.grad.clone())
                # torch.save(gradients, grad_checkpt_file)

                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break

          

        

    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    test_acc, metric_macro, metric_micro, roc = test(classifier, args.dev, test_loader,checkpt_file)
    
    print("Time of propagation", np.sum(change_node_list))
    print(f"Train cost: {train_time:.2f}s")
    print('Load {}th epoch'.format(best_epoch))
    print(f"Test accuracy:{100*test_acc:.2f}%")
    print(f"metric_macro:{100*metric_macro:.2f}%")
    print(f"metric_micro:{100*metric_micro:.2f}%")
    print(f"ROC:{100*roc:.2f}%")
    print(f"Memory: {memory / 2**20:.3f}GB")
    table_accuracy_path = Path('/home/ubuntu/project/pytorch_geometric_temporal/examples/log/'+args.dataset.lower()+'_accuracy_table.csv')
    table_efficiency_path = Path('/home/ubuntu/project/pytorch_geometric_temporal/examples/log/'+args.dataset.lower()+'_efficiency_table.csv')
    table_memory_path = Path('/home/ubuntu/project/pytorch_geometric_temporal/examples/log/'+args.dataset.lower()+'_memory_table.csv')
    update_results('cute', snapshot,100*test_acc,table_accuracy_path)
    update_results('cute', snapshot,(np.sum(change_node_list)+train_time)/args.epochs,table_efficiency_path)
    update_results('cute', snapshot,memory / 2**20,table_memory_path)

    return metric_macro, metric_micro

if __name__ == '__main__':
    main()
