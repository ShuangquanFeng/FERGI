import os
import json
import numpy as np
import pandas as pd
import pickle as pkl
from collections import defaultdict
import itertools
import random
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from FAUNet import *
from config import *

parser = argparse.ArgumentParser(description='Classify image preferences')
parser.add_argument('--seed', default=0, type=int, help='Random seed.')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--epochs', default=300, type=int, help="number of epochs for training")
parser.add_argument('--print_interval', default=100, type=int, help='define the interval of epochs to print results')

l1_lambda = 0.0001
layer_sizes = [12, 16, 1]

def main():
    args = parser.parse_args()
    seed = args.seed
    device = args.gpu
    num_epochs = args.epochs
    print_interval = args.print_interval

    if seed is not None:
        # Make the model training deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    
    # Load the data
    dfs = []
    for user_id in user_id_list:
        df_reaction = pd.read_csv(os.path.join(prep_rootdir, 'preprocessed_reaction_data', f"{user_id}.csv"))
        df_images = pd.read_csv(os.path.join(prep_rootdir, 'preprocessed_image_data', f"{user_id}.csv"))
        df = pd.merge(df_reaction, df_images, on = ['session_index', 'image_index'], how='inner')
        df['user_id'] = user_id
        dfs.append(df)
    df_reaction_all = pd.concat(dfs)
    
    data_dict = defaultdict(list)
    data_dict['user_id'] = user_id_list
    
    
    AU_cols = [col for col in df_reaction_all.columns if col.startswith('AU')]
    df_metrics_data_1, df_metrics_data_2 = get_data_pair(df_reaction_all, ['rank_score'] + AU_cols)


    inputs_1 = df_metrics_data_1[AU_cols].values
    inputs_2 = df_metrics_data_2[AU_cols].values
    
    inputs_1 = torch.tensor(inputs_1, dtype=torch.float32).to(device)
    inputs_2 = torch.tensor(inputs_2, dtype=torch.float32).to(device)
    
    targets = (df_metrics_data_1['rank_score'].values > df_metrics_data_2['rank_score'].values).astype(float)
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Define the model
    preprocess_steps = ['linear', 'sigmoid']
    activation_function = 'sigmoid'
    
    model = FAUNet(preprocess=preprocess_steps, layer_sizes=layer_sizes, activation=activation_function)
    model.to(device)
    
    criterion = RankNetLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs_1 = model(inputs_1)
        outputs_2 = model(inputs_2)
        
        loss = criterion(outputs_1, outputs_2, targets)

        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm
        
        loss.backward()
        optimizer.step()
      
    torch.save(model.state_dict(), os.path.join(results_rootdir, 'FAU-Net.pth'))

def get_data_pair(df_reaction, metrics):
    all_metrics_data_list1 = defaultdict(list)
    all_metrics_data_list2 = defaultdict(list)
    for user_id in np.unique(df_reaction['user_id']):
        df_participant = df_reaction[df_reaction['user_id'] == user_id].copy()
    
        participant_scores_pairwise_diff = []
        participant_ranks_pairwise_diff = []
        for session in np.unique(df_participant['session_index']):
            df_session = df_participant[df_participant['session_index'] == session]
    
            metric_data_list1 = {}
            metric_data_list2 = {}
            for metric in metrics:
                scores = df_session[metric].to_numpy()
                scores_list1 = np.tile(scores[None,:], [scores.size, 1]).flatten()
                scores_list2 = np.tile(scores[:,None], [1, scores.size]).flatten()
                metric_data_list1[metric] = scores_list1
                metric_data_list2[metric] = scores_list2
    
            compare_idx = np.where(metric_data_list1['rank_score'] != metric_data_list2['rank_score'])[0]
            for metric in metrics:
                all_metrics_data_list1[metric].extend(metric_data_list1[metric][compare_idx])
                all_metrics_data_list2[metric].extend(metric_data_list2[metric][compare_idx])
    
    df_metrics_data_1 = pd.DataFrame(dict(all_metrics_data_list1))
    df_metrics_data_2 = pd.DataFrame(dict(all_metrics_data_list2))
    return df_metrics_data_1, df_metrics_data_2

class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()
    
    def forward(self, output1, output2, target):
        Pij = nn.Sigmoid()(output1 - output2)
        loss = nn.BCELoss()(Pij, target)
        return loss

if __name__ == '__main__':
    main()