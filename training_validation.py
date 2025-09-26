import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

from models.pdc_ginconv import PDC_GINConvNet
from models.vnoc_ginconv import Vnoc_GINConvNet
from models.pdc_vnoc_ginconv import PDC_Vnoc_GINConvNet
from models.esm_ginconv import ESM_GINConvNet
from models.fri_ginconv import FRI_GINConvNet

from models.esm_gat import ESM_GATNet

import wandb
import random
from utils import *
import argparse
from tqdm import tqdm

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, wandb_log=False):
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc=f"Epoch {epoch}"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        # if batch_idx % LOG_INTERVAL == 0:
            # print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
            #                                                                batch_idx * len(data.x),
            #                                                                len(train_loader.dataset),
            #                                                                100. * batch_idx / len(train_loader),
            #                                                                loss.item()))
    tqdm.write('Train loss: {:.6f}'.format(loss.item()))
    if wandb_log:
        wandb.log({"loss": loss.item()}, commit=False)

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    # print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), leave=False, desc="Predicting"):
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

datasets = ['davis', 'kiba']

all_models = {
    'GINConvNet': GINConvNet, 
    'GATNet': GATNet, 
    'GAT_GCN': GAT_GCN, 
    'GCNNet': GCNNet, 
    'PDC_GINConvNet': PDC_GINConvNet, 
    'Vnoc_GINConvNet': Vnoc_GINConvNet, 
    'ESM_GINConvNet': ESM_GINConvNet, 
    'FRI_GINConvNet': FRI_GINConvNet, 
    'PDC_Vnoc_GINConvNet': PDC_Vnoc_GINConvNet,
    'ESM_GATNet': ESM_GATNet
}

parser = argparse.ArgumentParser(description="Run a specific model on a specific dataset.")

parser.add_argument('--dataset', type=str, choices=datasets, required=True, 
                    help="Dataset name: 'davis' or 'kiba'.")
parser.add_argument('--model', type=str, choices=list(all_models.keys()), required=True, 
                    help="Model name. Choose from: " + ", ".join(all_models.keys()) + ".")
parser.add_argument('--cuda', type=int, default=0, 
                    help="CUDA device index (default: 0).")
parser.add_argument('--seed', type=int, default=None,
                    help="Random seed for reproducibility (default: None).")
parser.add_argument('--wandb', action='store_true', default=False,
                    help="Flag for using wandb logging (default: False).")
parser.add_argument('-vf', '--validation_fold', type=int, default=0,
                    help="Fold index to use for validation when using k-fold cross-validation (default: 0).")
# parser.add_argument('--split_type', type=str, default=None,
#                     help="Type of data split. Choose from: 'random', 'original', 'kfold', 'protein_cold', 'drug_cold', or 'fully_cold'.")
parser.add_argument('--mutation', action='store_true', default=False,
                    help="Flag for including protein sequence mutations for the Davis dataset (default: False).")

args = parser.parse_args()

modeling = all_models[args.model]
model_st = modeling.__name__

if model_st in ["ESM_GINConvNet", "ESM_GATNet"]:
    target_type = 'esm'
elif model_st == "FRI_GINConvNet":
    target_type = 'deepfri'
else:
    target_type = None

dataset = args.dataset
# split_type = args.split_type

# Select CUDA device if applicable
cuda_name = f"cuda:{args.cuda}"
print('cuda_name:', cuda_name)

# Set seed:
if args.seed is not None:
    seed = args.seed
    print("Seed: " + str(seed))
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

group_name = f"{args.model}_{args.dataset}"
run_name = f"{args.model}_{args.dataset}"
if args.mutation:
    group_name += "_mutation"
    run_name += "_mutation"
if args.seed is not None:
    run_name += f"_seed_{args.seed}"
run_name += f"_fold_{args.validation_fold}"

if args.wandb:
    wandb.init(project = 'GraphDTA', config = args, group = group_name, name = run_name )
    
# Main program: Train on specified dataset 
if __name__ == "__main__":
    print('Training ' + model_st + ' on ' + dataset + ' dataset...')
    dta_dataset = DTADataset(root='data', dataset=dataset, target_type=target_type, mutation=args.mutation)

    # original k-fold split (hard coded!)
    all_folds = [0, 1, 2, 3, 4]
    val_fold = args.validation_fold
    train_folds = [f for f in all_folds if f != val_fold]
    
    train_mask = torch.isin(dta_dataset._data.fold, torch.tensor(train_folds))
    train_dataset = dta_dataset[train_mask]
    val_dataset = dta_dataset[dta_dataset._data.fold == val_fold]
    test_dataset = dta_dataset[dta_dataset._data.fold == -1]

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_mse = 1000
    best_epoch = -1

    for epoch in range(NUM_EPOCHS):
        tqdm.write(f'\nEpoch {epoch+1}')
        train(model, device, train_loader, optimizer, epoch+1, wandb_log=args.wandb)

        G,P = predicting(model, device, val_loader)
        ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]
        if args.wandb:
            wandb.log({"rmse": ret[0], "mse": ret[1], "pearson": ret[2], "spearman": ret[3]})

        if ret[1]<best_mse:
            best_model = model
            best_epoch = epoch+1
            best_mse = ret[1]
        tqdm.write(f'Validation MSE: {ret[1]:.6f}\nBest MSE: {best_mse:.6f} (epoch {best_epoch})')

    model_file_name = 'trained_models/model_' + model_st + '_' + dataset + '_' + str(val_fold) + '_validation.model'
    result_file_name = 'trained_models/result_' + model_st + '_' + dataset + '_' + str(val_fold) + '_validation.csv'
    os.makedirs('trained_models', exist_ok=True)

    torch.save(best_model.state_dict(), model_file_name)

    G,P = predicting(best_model, device, val_loader)
    val_ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]

    tqdm.write('\nResults on val set:')
    tqdm.write(f"RMSE: {val_ret[0]}")
    tqdm.write(f"MSE: {val_ret[1]}")
    tqdm.write(f"Pearson: {val_ret[2]}")
    tqdm.write(f"Spearman: {val_ret[3]}")
    tqdm.write(f"CI: {val_ret[4]}\n")

    G,P = predicting(best_model, device, test_loader)
    test_ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]

    tqdm.write('\nResults on test set:')
    tqdm.write(f"RMSE: {test_ret[0]}")
    tqdm.write(f"MSE: {test_ret[1]}")
    tqdm.write(f"Pearson: {test_ret[2]}")
    tqdm.write(f"Spearman: {test_ret[3]}")
    tqdm.write(f"CI: {test_ret[4]}")

    if args.wandb:
        wandb.log({
            "test_rmse": test_ret[0],
            "test_mse": test_ret[1],
            "test_pearson": test_ret[2],
            "test_spearman": test_ret[3],
            "test_ci": test_ret[4]
        })
        wandb.finish()

    with open(result_file_name, 'w') as f:
        # write header
        f.write("test_rmse,test_mse,test_pearson,test_spearman,test_ci,val_rmse,val_mse,val_pearson,val_spearman,val_ci\n")
        # write values
        f.write(','.join(map(str, test_ret)))
        f.write(',')
        f.write(','.join(map(str, val_ret)))
        f.write('\n')
    