import os
import json
import pickle as pkl
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import defaultdict
import itertools
from config import *

def main():
    # Exclude participants with systematically unreliable, unstable AU4 estimation
    with open(os.path.join(prep_rootdir, 'AU4_excluded_participants.json'), 'r') as f:
        excluded_participants = json.load(f)
    filtered_user_id_list = [p for p in user_id_list if p not in excluded_participants]
    
    # Load the data
    dfs = []
    for user_id in filtered_user_id_list:
        df_reaction = pd.read_csv(os.path.join(prep_rootdir, 'preprocessed_reaction_data', f"{user_id}.csv"))
        df_images = pd.read_csv(os.path.join(prep_rootdir, 'preprocessed_image_data', f"{user_id}.csv"))
        df = pd.merge(df_reaction, df_images, on = ['session_index', 'image_index'], how='inner')
        df['user_id'] = user_id
        dfs.append(df)
    df_reaction_all = pd.concat(dfs)
    
    data_dict = defaultdict(list)
    data_dict['user_id'] = filtered_user_id_list
    
    basic_metrics = ['CLIP_score', 'Aesthetic_score', 'BLIP_score', 'ImageReward_score', 'PickScore', 'HPS_V2_score']
    
    # Ensemble of the three pre-trained scoring models
    granularity = 0.1
    weights = np.arange(0, 1 + granularity, granularity)
    all_weight_combinations = [[w1, w2, 1 - w1 - w2] for w1 in weights for w2 in weights if (w1 + w2) <= 1]
    all_weight_combinations = np.array([combo for combo in all_weight_combinations if combo[2] >= 0])
    for left_out_user_id in filtered_user_id_list:
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
        
            
    # Image preference classification based on AUcomb valence score on only the selected data
    exp_decay_coef_list = np.arange(0+0.1, 2.1+0.1, 0.1)
    AU4_weight_list = np.arange(0, 1 + 0.1, 0.1)
    diff_threshold_list = np.arange(0.0, 1.0+0.02, 0.02)
    AUcomb_sel_classification_coefs_list = np.array(list(itertools.product(exp_decay_coef_list, AU4_weight_list, diff_threshold_list)))
    
    for user_id_idx, left_out_user_id in enumerate(filtered_user_id_list):
        df_reaction_train = df_reaction_all[df_reaction_all['user_id'] != left_out_user_id].copy()
        df_reaction_test = df_reaction_all[df_reaction_all['user_id'] == left_out_user_id].copy()
        
        for metric in basic_metrics:
            train_mean, train_std = df_reaction_train[metric].mean(), df_reaction_train[metric].std()
            df_reaction_train[metric] = (df_reaction_train[metric] - train_mean) / train_std
            df_reaction_test[metric] = (df_reaction_test[metric] - train_mean) / train_std
            
        df_reaction_train['ensemble_baseline_score'] = np.vectorize(lambda u: data_dict['ImageReward_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['ImageReward_score'].to_numpy() + np.vectorize(lambda u: data_dict['PickScore_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['PickScore'].to_numpy() + np.vectorize(lambda u: data_dict['HPS_V2_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['HPS_V2_score'].to_numpy()
        
        df_reaction_test['ensemble_baseline_score'] = np.vectorize(lambda u: data_dict['ImageReward_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['ImageReward_score'].to_numpy() + np.vectorize(lambda u: data_dict['PickScore_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['PickScore'].to_numpy() + np.vectorize(lambda u: data_dict['HPS_V2_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['HPS_V2_score'].to_numpy()
        
        df_train_metrics_data_1, df_train_metrics_data_2 = get_data_pair(df_reaction_train, basic_metrics + ['rank_score', 'ensemble_baseline_score', 'AU4_activation_value', 'AU12_activation_value'])
        df_test_metrics_data_1, df_test_metrics_data_2 = get_data_pair(df_reaction_test, basic_metrics + ['rank_score', 'ensemble_baseline_score', 'AU4_activation_value', 'AU12_activation_value'])

        def AUcomb_valence_scores_correctness(df_metrics_data_1, df_metrics_data_2, coefs):
            n_combinations = coefs.shape[0]
            exp_decay_coefs = coefs[:,0:1]
            AU4_coefs = coefs[:,1:2]
            AU12_coefs = 1 - AU4_coefs
            diff_thresholds = coefs[:,2:3]
            
            AU4_scores = - (1 - np.exp(-exp_decay_coefs * np.tile(df_metrics_data_1['AU4_activation_value'].to_numpy(), [n_combinations, 1])))
            AU12_scores = (1 - np.exp(-exp_decay_coefs * np.tile(df_metrics_data_1['AU12_activation_value'].to_numpy(), [n_combinations, 1])))
            data1_scores = AU4_coefs * AU4_scores + AU12_coefs * AU12_scores

            AU4_scores = - (1 - np.exp(-exp_decay_coefs * np.tile(df_metrics_data_2['AU4_activation_value'].to_numpy(), [n_combinations, 1])))
            AU12_scores = (1 - np.exp(-exp_decay_coefs * np.tile(df_metrics_data_2['AU12_activation_value'].to_numpy(), [n_combinations, 1])))
            data2_scores = AU4_coefs * AU4_scores + AU12_coefs * AU12_scores

            scores_pairwise_diff = data1_scores - data2_scores
            ranks_pairwise_diff = df_metrics_data_1['rank_score'].to_numpy() - df_metrics_data_2['rank_score'].to_numpy()

            compare_idx = np.abs(scores_pairwise_diff) >= diff_thresholds

            all_metrics_n_correct = {}
            all_metrics_n_incorrect = {}
            for metric in (basic_metrics + ['ensemble_baseline_score', 'AUcomb_valence_score']):
                if metric == 'AUcomb_valence_score':
                    metric_scores_pairwise_diff = scores_pairwise_diff
                    correctness = ranks_pairwise_diff * metric_scores_pairwise_diff > 0
                    n_correct, n_incorrect = np.sum(np.logical_and(correctness, compare_idx), axis=1) // 2, np.sum(np.logical_and(~correctness, compare_idx), axis=1) // 2

                else:
                    metric_scores_pairwise_diff = (df_metrics_data_1[metric] - df_metrics_data_2[metric]).to_numpy()
                    correctness = ranks_pairwise_diff * metric_scores_pairwise_diff > 0
                    n_correct, n_incorrect = np.sum(np.logical_and(np.tile(correctness, [n_combinations, 1]), compare_idx), axis=1) // 2, np.sum(np.logical_and(np.tile(~correctness, [n_combinations, 1]), compare_idx), axis=1) // 2

                all_metrics_n_correct[metric] = n_correct
                all_metrics_n_incorrect[metric] = n_incorrect

            return all_metrics_n_correct, all_metrics_n_incorrect
        
        all_metrics_train_n_correct, all_metrics_train_n_incorrect = AUcomb_valence_scores_correctness(df_train_metrics_data_1, df_train_metrics_data_2, AUcomb_sel_classification_coefs_list)
        all_metrics_test_n_correct, all_metrics_test_n_incorrect = AUcomb_valence_scores_correctness(df_test_metrics_data_1, df_test_metrics_data_2, AUcomb_sel_classification_coefs_list)

        best_idx = np.argmax(all_metrics_train_n_correct['AUcomb_valence_score'] - 2 * all_metrics_train_n_incorrect['AUcomb_valence_score'])
        best_exp_decay, best_AU4_coef, best_diff_threshold = AUcomb_sel_classification_coefs_list[best_idx]
        best_AU12_coef = 1 - best_AU4_coef
        
        data_dict['exp_decay_coef'].append(best_exp_decay)
        data_dict['AU4_coef'].append(best_AU4_coef)
        data_dict['AU12_coef'].append(best_AU12_coef)
        data_dict['diff_threshold'].append(best_diff_threshold)

        for metric in (basic_metrics + ['ensemble_baseline_score', 'AUcomb_valence_score']):
            data_dict[f"AUcomb_selected_{metric}_n_correct"].append(all_metrics_test_n_correct[metric][best_idx])
            data_dict[f"AUcomb_selected_{metric}_n_incorrect"].append(all_metrics_test_n_incorrect[metric][best_idx])

        print(f"AUcomb valence score optimization... User {left_out_user_id} complete...")
    
    df_reaction_all['AU4_valence_score'] = - (1 - np.exp(-np.vectorize(lambda u: data_dict['exp_decay_coef'][data_dict['user_id'].index(u)])(df_reaction_all['user_id']) * df_reaction_all['AU4_activation_value'].to_numpy()))
    df_reaction_all['AU12_valence_score'] = (1 - np.exp(-np.vectorize(lambda u: data_dict['exp_decay_coef'][data_dict['user_id'].index(u)])(df_reaction_all['user_id']) * df_reaction_all['AU12_activation_value'].to_numpy()))
    
    df_reaction_all['AUcomb_valence_score'] = np.vectorize(lambda u: data_dict['AU4_coef'][data_dict['user_id'].index(u)])(df_reaction_all['user_id']) * df_reaction_all['AU12_valence_score'] * df_reaction_all['AU4_valence_score'] + np.vectorize(lambda u: data_dict['AU12_coef'][data_dict['user_id'].index(u)])(df_reaction_all['user_id']) * df_reaction_all['AU12_valence_score']
    
    
    # Compute the performance of all scores
    for user_id in filtered_user_id_list:
        df_participant = df_reaction_all[df_reaction_all['user_id'] == user_id].copy()

        participant_scores_pairwise_diff = defaultdict(list)
        for session in np.unique(df_participant['session_index']):
            df_session = df_participant[df_participant['session_index'] == session].copy()

            scores_pairwise_diff = {}
            for metric in (basic_metrics + ['rank_score', 'AUcomb_valence_score']):
                scores = df_session[metric].to_numpy()
                scores_pairwise_diff[metric] = (scores[None,:] - scores[:,None]).flatten()
            
            compare_idx = np.where(scores_pairwise_diff['rank_score'] != 0)[0]
            for metric in (basic_metrics + ['rank_score', 'AUcomb_valence_score']):
                scores_pairwise_diff[metric] = scores_pairwise_diff[metric][compare_idx]
                participant_scores_pairwise_diff[metric].extend(scores_pairwise_diff[metric])

        for metric in basic_metrics + ['AUcomb_valence_score']:
            correctness = np.array(participant_scores_pairwise_diff['rank_score']) * np.array(participant_scores_pairwise_diff[metric]) > 0
            n_correct, n_incorrect = np.sum(correctness) // 2, np.sum(~correctness) // 2
            data_dict[f"{metric}_n_correct"].append(n_correct)
            data_dict[f"{metric}_n_incorrect"].append(n_incorrect)
    
    
    
    # Integrate AUcomb valence score with the pre-trained scoring models
    AUcomb_coef_list = np.arange(0, 10.0+0.1, 0.1)
    
    for (baseline_score, ensemble_score) in [('ImageReward_score', 'ensemble_ImageReward_AUcomb_score'), ('PickScore', 'ensemble_PickScore_AUcomb_score'), ('HPS_V2_score', 'ensemble_HPS_V2_AUcomb_score'), ('ensemble_baseline_score', 'ensemble_all_score')]:
        for user_id_idx, left_out_user_id in enumerate(filtered_user_id_list):
            df_reaction_train = df_reaction_all[df_reaction_all['user_id'] != left_out_user_id].copy()
            df_reaction_test = df_reaction_all[df_reaction_all['user_id'] == left_out_user_id].copy()

            for metric in basic_metrics:
                train_mean, train_std = df_reaction_train[metric].mean(), df_reaction_train[metric].std()
                df_reaction_train[metric] = (df_reaction_train[metric] - train_mean) / train_std
                df_reaction_test[metric] = (df_reaction_test[metric] - train_mean) / train_std
                
            df_reaction_train['ensemble_baseline_score'] = np.vectorize(lambda u: data_dict['ImageReward_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['ImageReward_score'].to_numpy() + np.vectorize(lambda u: data_dict['PickScore_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['PickScore'].to_numpy() + np.vectorize(lambda u: data_dict['HPS_V2_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['HPS_V2_score'].to_numpy()

            df_reaction_test['ensemble_baseline_score'] = np.vectorize(lambda u: data_dict['ImageReward_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['ImageReward_score'].to_numpy() + np.vectorize(lambda u: data_dict['PickScore_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['PickScore'].to_numpy() + np.vectorize(lambda u: data_dict['HPS_V2_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['HPS_V2_score'].to_numpy()

            df_train_metrics_data_1, df_train_metrics_data_2 = get_data_pair(df_reaction_train, basic_metrics + ['rank_score', 'ensemble_baseline_score', 'AUcomb_valence_score'])
            df_test_metrics_data_1, df_test_metrics_data_2 = get_data_pair(df_reaction_test, basic_metrics + ['rank_score', 'ensemble_baseline_score', 'AUcomb_valence_score'])
    
            def ensemble_scores_correctness(df_metrics_data_1, df_metrics_data_2, weights):
                n_combinations = weights.shape[0]
                scores_pairwise_diff = (np.tile(df_metrics_data_1[baseline_score].to_numpy(), [n_combinations, 1]) + np.expand_dims(weights, 1) *  np.tile(df_metrics_data_1['AUcomb_valence_score'].to_numpy(), [n_combinations, 1])) - (np.tile(df_metrics_data_2[baseline_score].to_numpy(), [n_combinations, 1]) + np.expand_dims(weights, 1) *  np.tile(df_metrics_data_2['AUcomb_valence_score'].to_numpy(), [n_combinations, 1]))
                ranks_pairwise_diff = df_metrics_data_1['rank_score'].to_numpy() - df_metrics_data_2['rank_score'].to_numpy()
                correctness = ranks_pairwise_diff * scores_pairwise_diff > 0
                n_correct, n_incorrect = np.sum(correctness, axis=1) // 2, np.sum(~correctness, axis=1) // 2
                return n_correct, n_incorrect

            train_n_correct, train_n_incorrect = ensemble_scores_correctness(df_train_metrics_data_1, df_train_metrics_data_2, AUcomb_coef_list)
            test_n_correct, test_n_incorrect = ensemble_scores_correctness(df_test_metrics_data_1, df_test_metrics_data_2, AUcomb_coef_list)
    
            best_idx = np.argmax(train_n_correct)
            best_AUcomb_coef = AUcomb_coef_list[best_idx]
            
            data_dict[f'AUcomb_coef_for_{ensemble_score}'].append(best_AUcomb_coef)
            data_dict[f'{ensemble_score}_n_correct'].append(test_n_correct[best_idx])
            data_dict[f'{ensemble_score}_n_incorrect'].append(test_n_incorrect[best_idx])
    
            print(f"{ensemble_score} optimization... User {left_out_user_id} complete...")  
        
    df_results = pd.DataFrame(data_dict)
    os.makedirs(results_rootdir, exist_ok=True)
    df_results.to_csv(os.path.join(results_rootdir, 'image_preference_binary_classification_based_on_ranking.csv'), index=False)


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


if __name__ == '__main__':
    main()