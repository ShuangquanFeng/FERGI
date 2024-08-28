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
    
    basic_metrics = ['CLIP_score', 'Aesthetic_score', 'BLIP_score', 'ImageReward_score', 'PickScore', 'HPS_V2_score']
    
    # Ensemble of the three pre-trained scoring models
    granularity = 0.1
    weights = np.arange(0, 1 + granularity, granularity)
    all_weight_combinations = [[w1, w2, 1 - w1 - w2] for w1 in weights for w2 in weights if (w1 + w2) <= 1]
    all_weight_combinations = np.array([combo for combo in all_weight_combinations if combo[2] >= 0])
    for left_out_user_id in user_id_list:
        df_reaction_train = df_reaction_all[df_reaction_all['user_id'] != left_out_user_id].copy()
        df_reaction_test = df_reaction_all[df_reaction_all['user_id'] == left_out_user_id].copy()
        for metric in basic_metrics:
            train_mean, train_std = df_reaction_train[metric].mean(), df_reaction_train[metric].std()
            df_reaction_train[metric] = (df_reaction_train[metric] - train_mean) / train_std
            df_reaction_test[metric] = (df_reaction_test[metric] - train_mean) / train_std

        df_train_metrics_data_1, df_train_metrics_data_2 = get_data_pair(df_reaction_train, basic_metrics + ['rank_score'])
        df_test_metrics_data_1, df_test_metrics_data_2 = get_data_pair(df_reaction_test, basic_metrics + ['rank_score'])

        def ensemble_scores_correctness(df_metrics_data_1, df_metrics_data_2, weights):
            n_combinations = weights.shape[0]
            scores_pairwise_diff = ((weights[:,0:1] * np.tile(df_metrics_data_1['ImageReward_score'].to_numpy(), [n_combinations, 1])) + (weights[:,1:2] * np.tile(df_metrics_data_1['PickScore'].to_numpy(), [n_combinations, 1])) + (weights[:,2:3] * np.tile(df_metrics_data_1['HPS_V2_score'].to_numpy(), [n_combinations, 1]))) - ((weights[:,0:1] * np.tile(df_metrics_data_2['ImageReward_score'].to_numpy(), [n_combinations, 1])) + (weights[:,1:2] * np.tile(df_metrics_data_2['PickScore'].to_numpy(), [n_combinations, 1])) + (weights[:,2:3] * np.tile(df_metrics_data_2['HPS_V2_score'].to_numpy(), [n_combinations, 1])))
            ranks_pairwise_diff = df_metrics_data_1['rank_score'].to_numpy() - df_metrics_data_2['rank_score'].to_numpy()
            correctness = ranks_pairwise_diff * scores_pairwise_diff > 0
            n_correct, n_incorrect = np.sum(correctness, axis=1) // 2, np.sum(~correctness, axis=1) // 2
            return n_correct, n_incorrect

        train_n_correct, train_n_incorrect = ensemble_scores_correctness(df_train_metrics_data_1, df_train_metrics_data_2, all_weight_combinations)
        test_n_correct, test_n_incorrect = ensemble_scores_correctness(df_test_metrics_data_1, df_test_metrics_data_2, all_weight_combinations)

        best_idx = np.argmax(train_n_correct)
        best_w1, best_w2, best_w3 = all_weight_combinations[best_idx]
        
        data_dict['ImageReward_weight'].append(best_w1)
        data_dict['PickScore_weight'].append(best_w2)
        data_dict['HPS_V2_weight'].append(best_w3)

        data_dict['ensemble_baseline_score_n_correct'].append(test_n_correct[best_idx])
        data_dict['ensemble_baseline_score_n_incorrect'].append(test_n_incorrect[best_idx])

        print(f"Ensemble baseline score optimization... User {left_out_user_id} complete...")

    sel_n_correct = {}
    sel_n_incorrect = {}
    for sel_metric in basic_metrics + ['ensemble_baseline_score', 'AUs_integrated_score']:
        sel_n_correct[sel_metric] = {}
        sel_n_incorrect[sel_metric] = {}
        for metric in basic_metrics + ['ensemble_baseline_score', 'AUs_integrated_score']:
            sel_n_correct[sel_metric][metric] = defaultdict(int)
            sel_n_incorrect[sel_metric][metric] = defaultdict(int)
    all_AUs_integrated_scores = []
    for user_id_idx, left_out_user_id in enumerate(user_id_list):
        df_reaction_train = df_reaction_all[df_reaction_all['user_id'] != left_out_user_id].copy()
        df_reaction_test = df_reaction_all[df_reaction_all['user_id'] == left_out_user_id].copy()
        
        for metric in basic_metrics:
            train_mean, train_std = df_reaction_train[metric].mean(), df_reaction_train[metric].std()
            df_reaction_train[metric] = (df_reaction_train[metric] - train_mean) / train_std
            df_reaction_test[metric] = (df_reaction_test[metric] - train_mean) / train_std
            
        df_reaction_train['ensemble_baseline_score'] = np.vectorize(lambda u: data_dict['ImageReward_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['ImageReward_score'].to_numpy() + np.vectorize(lambda u: data_dict['PickScore_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['PickScore'].to_numpy() + np.vectorize(lambda u: data_dict['HPS_V2_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['HPS_V2_score'].to_numpy()
        
        df_reaction_test['ensemble_baseline_score'] = np.vectorize(lambda u: data_dict['ImageReward_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['ImageReward_score'].to_numpy() + np.vectorize(lambda u: data_dict['PickScore_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['PickScore'].to_numpy() + np.vectorize(lambda u: data_dict['HPS_V2_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['HPS_V2_score'].to_numpy()
        
        AU_cols = [col for col in df_reaction_train.columns if col.startswith('AU')]
        df_train_metrics_data_1, df_train_metrics_data_2 = get_data_pair(df_reaction_train, basic_metrics + ['rank_score', 'ensemble_baseline_score'] + AU_cols)
        df_test_metrics_data_1, df_test_metrics_data_2 = get_data_pair(df_reaction_test, basic_metrics + ['rank_score', 'ensemble_baseline_score'] + AU_cols)


        print(f"Participant {left_out_user_id} started...")
        # Prepare training inputs and targets
        train_inputs_1 = df_train_metrics_data_1[AU_cols].values
        train_inputs_2 = df_train_metrics_data_2[AU_cols].values
        
        train_inputs_1 = torch.tensor(train_inputs_1, dtype=torch.float32).to(device)
        train_inputs_2 = torch.tensor(train_inputs_2, dtype=torch.float32).to(device)
        
        train_targets = (df_train_metrics_data_1['rank_score'].values > df_train_metrics_data_2['rank_score'].values).astype(float)
        train_targets = torch.tensor(train_targets, dtype=torch.float32).unsqueeze(1).to(device)
        
        # Prepare test inputs and targets
        test_inputs_1 = df_test_metrics_data_1[AU_cols].values
        test_inputs_2 = df_test_metrics_data_2[AU_cols].values
        
        test_inputs_1 = torch.tensor(test_inputs_1, dtype=torch.float32).to(device)
        test_inputs_2 = torch.tensor(test_inputs_2, dtype=torch.float32).to(device)
        
        test_targets = (df_test_metrics_data_1['rank_score'].values > df_test_metrics_data_2['rank_score'].values).astype(float)
        test_targets = torch.tensor(test_targets, dtype=torch.float32).unsqueeze(1).to(device)
        
        # Define the model
        preprocess_steps = ['linear', 'sigmoid']
        activation_function = 'sigmoid'
        
        model = FAUNet(preprocess=preprocess_steps, layer_sizes=layer_sizes, activation=activation_function)
        model.to(device)
        
        # Define loss function and optimizer
        criterion = RankNetLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs_1 = model(train_inputs_1)
            outputs_2 = model(train_inputs_2)
            
            loss = criterion(outputs_1, outputs_2, train_targets)

            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % print_interval == 0:
                model.eval()
                with torch.no_grad():
                    test_predictions_1 = model(test_inputs_1)
                    test_predictions_2 = model(test_inputs_2)
                    test_predicted_labels = (test_predictions_1 > test_predictions_2).float()
                    test_accuracy = (test_predicted_labels == test_targets).float().mean()
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}')
                
        # Final model evaluation
        model.eval()
        with torch.no_grad():
            test_predictions_1 = model(test_inputs_1).cpu().numpy()
            test_predictions_2 = model(test_inputs_2).cpu().numpy()
            AUs_integrated_scores_diff = test_predictions_1 - test_predictions_2
            test_predicted_labels = (AUs_integrated_scores_diff > 0)
            test_targets = test_targets.cpu().numpy()
            correctness = (test_predicted_labels == test_targets)
            n_correct, n_incorrect = np.sum(correctness) // 2, np.sum(~correctness) // 2
            test_accuracy = n_correct / (n_correct + n_incorrect)
            print(f'Final Test Accuracy: {test_accuracy:.4f}')

            all_AUs_integrated_scores.extend(model(torch.tensor(df_reaction_test[AU_cols].values, dtype=torch.float32).to(device)).cpu().numpy()[:,0].tolist())
        print(f"Participant {left_out_user_id} finished...")

        data_dict[f"AUs_integrated_score_n_correct"].append(n_correct)
        data_dict[f"AUs_integrated_score_n_incorrect"].append(n_incorrect)

        
        for proportion in np.linspace(0.01, 1.00, 100):
            for sel_metric in (basic_metrics + ['ensemble_baseline_score', 'AUs_integrated_score']):
                if sel_metric != 'AUs_integrated_score':
                    scores_diff = df_test_metrics_data_1[sel_metric] - df_test_metrics_data_2[sel_metric]
                    sel_idx = np.abs(scores_diff) >= np.percentile(np.abs(scores_diff), 100*(1-proportion))
                else:
                    sel_idx = np.abs(AUs_integrated_scores_diff[:,0]) >= np.percentile(np.abs(AUs_integrated_scores_diff[:,0]), 100*(1-proportion))
                for metric in (basic_metrics + ['ensemble_baseline_score']):
                    scores_diff = df_test_metrics_data_1[metric] - df_test_metrics_data_2[metric]
                    scores_diff = scores_diff[sel_idx]
                    predicted_labels = scores_diff > 0
                    correctness = (predicted_labels == test_targets[sel_idx,0])
                    n_correct, n_incorrect = np.sum(correctness), np.sum(~correctness)
                    sel_n_correct[sel_metric][metric][proportion] += n_correct
                    sel_n_incorrect[sel_metric][metric][proportion] += n_incorrect
                predicted_labels = AUs_integrated_scores_diff[sel_idx,0] > 0
                correctness = (predicted_labels == test_targets[sel_idx,0])
                n_correct, n_incorrect = np.sum(correctness), np.sum(~correctness)
                sel_n_correct[sel_metric]['AUs_integrated_score'][proportion] += n_correct
                sel_n_incorrect[sel_metric]['AUs_integrated_score'][proportion] += n_incorrect


        print(f"AUs integrated score optimization... User {left_out_user_id} complete...")

    
    with open(os.path.join(results_rootdir, 'sel_correctness.pkl'), "wb") as f:
        pkl.dump({'sel_n_correct': sel_n_correct, 'sel_n_incorrect': sel_n_incorrect}, f)
    df_reaction_all['AUs_integrated_score'] = all_AUs_integrated_scores
    # import pdb; pdb.set_trace()
    
    
    # Compute the performance of all scores
    for user_id in user_id_list:
        df_participant = df_reaction_all[df_reaction_all['user_id'] == user_id].copy()

        participant_scores_pairwise_diff = defaultdict(list)
        for session in np.unique(df_participant['session_index']):
            df_session = df_participant[df_participant['session_index'] == session].copy()

            scores_pairwise_diff = {}
            for metric in (basic_metrics + ['rank_score']):
                scores = df_session[metric].to_numpy()
                scores_pairwise_diff[metric] = (scores[None,:] - scores[:,None]).flatten()
            
            compare_idx = np.where(scores_pairwise_diff['rank_score'] != 0)[0]
            for metric in (basic_metrics + ['rank_score']):
                scores_pairwise_diff[metric] = scores_pairwise_diff[metric][compare_idx]
                participant_scores_pairwise_diff[metric].extend(scores_pairwise_diff[metric])

        for metric in basic_metrics:
            correctness = np.array(participant_scores_pairwise_diff['rank_score']) * np.array(participant_scores_pairwise_diff[metric]) > 0
            n_correct, n_incorrect = np.sum(correctness) // 2, np.sum(~correctness) // 2
            data_dict[f"{metric}_n_correct"].append(n_correct)
            data_dict[f"{metric}_n_incorrect"].append(n_incorrect)
    
    
    
    # Integrate AUs integrated score with the pre-trained scoring models
    AUs_coef_list = np.arange(0, 10.0+0.1, 0.1)
    
    for (baseline_score, ensemble_score) in [('ImageReward_score', 'ensemble_ImageReward_AUs_score'), ('PickScore', 'ensemble_PickScore_AUs_score'), ('HPS_V2_score', 'ensemble_HPS_V2_AUs_score'), ('ensemble_baseline_score', 'ensemble_all_score')]:
        for user_id_idx, left_out_user_id in enumerate(user_id_list):
            df_reaction_train = df_reaction_all[df_reaction_all['user_id'] != left_out_user_id].copy()
            df_reaction_test = df_reaction_all[df_reaction_all['user_id'] == left_out_user_id].copy()

            for metric in basic_metrics:
                train_mean, train_std = df_reaction_train[metric].mean(), df_reaction_train[metric].std()
                df_reaction_train[metric] = (df_reaction_train[metric] - train_mean) / train_std
                df_reaction_test[metric] = (df_reaction_test[metric] - train_mean) / train_std
                
            df_reaction_train['ensemble_baseline_score'] = np.vectorize(lambda u: data_dict['ImageReward_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['ImageReward_score'].to_numpy() + np.vectorize(lambda u: data_dict['PickScore_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['PickScore'].to_numpy() + np.vectorize(lambda u: data_dict['HPS_V2_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['HPS_V2_score'].to_numpy()

            df_reaction_test['ensemble_baseline_score'] = np.vectorize(lambda u: data_dict['ImageReward_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['ImageReward_score'].to_numpy() + np.vectorize(lambda u: data_dict['PickScore_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['PickScore'].to_numpy() + np.vectorize(lambda u: data_dict['HPS_V2_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['HPS_V2_score'].to_numpy()

            df_train_metrics_data_1, df_train_metrics_data_2 = get_data_pair(df_reaction_train, basic_metrics + ['rank_score', 'ensemble_baseline_score', 'AUs_integrated_score'])
            df_test_metrics_data_1, df_test_metrics_data_2 = get_data_pair(df_reaction_test, basic_metrics + ['rank_score', 'ensemble_baseline_score', 'AUs_integrated_score'])
    
            def ensemble_scores_correctness(df_metrics_data_1, df_metrics_data_2, weights):
                n_combinations = weights.shape[0]
                scores_pairwise_diff = (np.tile(df_metrics_data_1[baseline_score].to_numpy(), [n_combinations, 1]) + np.expand_dims(weights, 1) *  np.tile(df_metrics_data_1['AUs_integrated_score'].to_numpy(), [n_combinations, 1])) - (np.tile(df_metrics_data_2[baseline_score].to_numpy(), [n_combinations, 1]) + np.expand_dims(weights, 1) *  np.tile(df_metrics_data_2['AUs_integrated_score'].to_numpy(), [n_combinations, 1]))
                ranks_pairwise_diff = df_metrics_data_1['rank_score'].to_numpy() - df_metrics_data_2['rank_score'].to_numpy()
                correctness = ranks_pairwise_diff * scores_pairwise_diff > 0
                n_correct, n_incorrect = np.sum(correctness, axis=1) // 2, np.sum(~correctness, axis=1) // 2
                return n_correct, n_incorrect

            train_n_correct, train_n_incorrect = ensemble_scores_correctness(df_train_metrics_data_1, df_train_metrics_data_2, AUs_coef_list)
            test_n_correct, test_n_incorrect = ensemble_scores_correctness(df_test_metrics_data_1, df_test_metrics_data_2, AUs_coef_list)
    
            best_idx = np.argmax(train_n_correct)
            best_AUs_coef = AUs_coef_list[best_idx]
            
            data_dict[f'AUs_coef_for_{ensemble_score}'].append(best_AUs_coef)
            data_dict[f'{ensemble_score}_n_correct'].append(test_n_correct[best_idx])
            data_dict[f'{ensemble_score}_n_incorrect'].append(test_n_incorrect[best_idx])
    
            print(f"{ensemble_score} optimization... User {left_out_user_id} complete...")
 
        
    df_results = pd.DataFrame(data_dict)
    os.makedirs(results_rootdir, exist_ok=True)
    df_results.to_csv(os.path.join(results_rootdir, 'image_preference_binary_classification_based_on_ranking_NN.csv'), index=False)
    



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