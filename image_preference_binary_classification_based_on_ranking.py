import os
import json
import pickle as pkl
import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
from config import *

def main():
    # Exclude participants with systematically unreliable, unstable AU4 estimatio
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
    all_weight_combinations = [(w1, w2, 1 - w1 - w2) for w1 in weights for w2 in weights if (w1 + w2) <= 1]
    all_weight_combinations = [combo for combo in all_weight_combinations if combo[2] >= 0]
    for left_out_user_id in filtered_user_id_list:
        df_reaction_train = df_reaction_all[df_reaction_all['user_id'] != left_out_user_id].copy()
        df_reaction_test = df_reaction_all[df_reaction_all['user_id'] == left_out_user_id].copy()
        for metric in basic_metrics:
            train_mean, train_std = df_reaction_train[metric].mean(), df_reaction_train[metric].std()
            df_reaction_train[metric] = (df_reaction_train[metric] - train_mean) / train_std
            df_reaction_test[metric] = (df_reaction_test[metric] - train_mean) / train_std
        

        train_n_correct = {}
        train_n_incorrect = {}
        test_n_correct = {}
        test_n_incorrect = {}
        for ensemble_weights in all_weight_combinations:
            w1, w2, w3 = ensemble_weights
            
            train_rank_scores_pairwise_diff = []
            train_ensemble_scores_pairwise_diff = []
            test_rank_scores_pairwise_diff = []
            test_ensemble_scores_pairwise_diff = []
            
            for (user_id, session_index), df_session in df_reaction_train.groupby(['user_id', 'session_index']):
                rank_scores = df_session['rank_score'].to_numpy()
                
                s1 = df_session['ImageReward_score'].to_numpy()
                s2 = df_session['PickScore'].to_numpy()
                s3 = df_session['HPS_V2_score'].to_numpy()
                ensemble_scores = w1*s1 + w2*s2 + w3*s3

                rank_scores_pairwise_diff = (rank_scores[None,:] - rank_scores[:,None]).flatten()
                ensemble_scores_pairwise_diff = (ensemble_scores[None,:] - ensemble_scores[:,None]).flatten()
                
                compare_idx = np.where(rank_scores_pairwise_diff != 0)[0]
                rank_scores_pairwise_diff = rank_scores_pairwise_diff[compare_idx]
                ensemble_scores_pairwise_diff = ensemble_scores_pairwise_diff[compare_idx]
                
                train_rank_scores_pairwise_diff.extend(rank_scores_pairwise_diff)
                train_ensemble_scores_pairwise_diff.extend(ensemble_scores_pairwise_diff)
                
            for (user_id, session_index), df_session in df_reaction_test.groupby(['user_id', 'session_index']):
                rank_scores = df_session['rank_score'].to_numpy()
                
                s1 = df_session['ImageReward_score'].to_numpy()
                s2 = df_session['PickScore'].to_numpy()
                s3 = df_session['HPS_V2_score'].to_numpy()
                ensemble_scores = w1*s1 + w2*s2 + w3*s3

                rank_scores_pairwise_diff = (rank_scores[None,:] - rank_scores[:,None]).flatten()
                ensemble_scores_pairwise_diff = (ensemble_scores[None,:] - ensemble_scores[:,None]).flatten()
                
                compare_idx = np.where(rank_scores_pairwise_diff != 0)[0]
                rank_scores_pairwise_diff = rank_scores_pairwise_diff[compare_idx]
                ensemble_scores_pairwise_diff = ensemble_scores_pairwise_diff[compare_idx]
                
                test_rank_scores_pairwise_diff.extend(rank_scores_pairwise_diff)
                test_ensemble_scores_pairwise_diff.extend(ensemble_scores_pairwise_diff)
                
            train_correctness = np.array(train_rank_scores_pairwise_diff) * np.array(train_ensemble_scores_pairwise_diff) > 0
            n_correct, n_incorrect = np.sum(train_correctness), np.sum(~train_correctness)
            train_n_correct[ensemble_weights] = n_correct
            train_n_incorrect[ensemble_weights] = n_incorrect
            
            test_correctness = np.array(test_rank_scores_pairwise_diff) * np.array(test_ensemble_scores_pairwise_diff) > 0
            n_correct, n_incorrect = np.sum(test_correctness), np.sum(~test_correctness)
            test_n_correct[ensemble_weights] = n_correct
            test_n_incorrect[ensemble_weights] = n_incorrect
            
        best_ensemble_weights = max(all_weight_combinations, key=train_n_correct.get)
        best_w1, best_w2, best_w3 = best_ensemble_weights
        data_dict['ImageReward_weight'].append(best_w1)
        data_dict['PickScore_weight'].append(best_w2)
        data_dict['HPS_V2_weight'].append(best_w3)

        data_dict['ensemble_baseline_score_n_correct'].append(test_n_correct[best_ensemble_weights])
        data_dict['ensemble_baseline_score_n_incorrect'].append(test_n_incorrect[best_ensemble_weights])
        
            
    # Image preference classification based on only AU4 valence score on only the selected data
    exp_decay_coef_list = np.arange(0+0.1, 2.1+0.1, 0.1)
    diff_threshold_list = np.arange(0.0, 1.0+0.02, 0.02)
    AU4_sel_classification_coefs_list = list(itertools.product(exp_decay_coef_list, diff_threshold_list))

    for user_id_idx, left_out_user_id in enumerate(filtered_user_id_list):
        df_reaction_train = df_reaction_all[df_reaction_all['user_id'] != left_out_user_id].copy()
        df_reaction_test = df_reaction_all[df_reaction_all['user_id'] == left_out_user_id].copy()
        
        for metric in basic_metrics:
            train_mean, train_std = df_reaction_train[metric].mean(), df_reaction_train[metric].std()
            df_reaction_train[metric] = (df_reaction_train[metric] - train_mean) / train_std
            df_reaction_test[metric] = (df_reaction_test[metric] - train_mean) / train_std
            
        df_reaction_train['ensemble_baseline_score'] = np.vectorize(lambda u: data_dict['ImageReward_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['ImageReward_score'].to_numpy() + np.vectorize(lambda u: data_dict['PickScore_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['PickScore'].to_numpy() + np.vectorize(lambda u: data_dict['HPS_V2_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['HPS_V2_score'].to_numpy()
        
        df_reaction_test['ensemble_baseline_score'] = np.vectorize(lambda u: data_dict['ImageReward_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['ImageReward_score'].to_numpy() + np.vectorize(lambda u: data_dict['PickScore_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['PickScore'].to_numpy() + np.vectorize(lambda u: data_dict['HPS_V2_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['HPS_V2_score'].to_numpy()
        
        train_n_correct = defaultdict(dict)
        train_n_incorrect = defaultdict(dict)
        test_n_correct = defaultdict(dict)
        test_n_incorrect = defaultdict(dict)
        for AU4_sel_classification_coefs in AU4_sel_classification_coefs_list:
            exp_decay_coef, diff_threshold = AU4_sel_classification_coefs
            
            train_scores_pairwise_diff = defaultdict(list)
            test_scores_pairwise_diff = defaultdict(list)

            for (user_id, session_index), df_session in df_reaction_train.groupby(['user_id', 'session_index']):
                scores_pairwise_diff = {}
                for metric in (basic_metrics + ['rank_score', 'ensemble_baseline_score', 'AU4_valence_score']):
                    if metric == 'AU4_valence_score':
                        scores = - (1 - np.exp(-exp_decay_coef * df_session['AU4_activation_value'].to_numpy()))
                    else:
                        scores = df_session[metric].to_numpy()
                    scores_pairwise_diff[metric] = (scores[None,:] - scores[:,None]).flatten()

                compare_idx = np.where(np.logical_and(scores_pairwise_diff['rank_score'] != 0, np.abs(scores_pairwise_diff['AU4_valence_score']) >= diff_threshold))[0]
                for metric in (basic_metrics + ['rank_score', 'ensemble_baseline_score', 'AU4_valence_score']):
                    scores_pairwise_diff[metric] = scores_pairwise_diff[metric][compare_idx]
                    train_scores_pairwise_diff[metric].extend(scores_pairwise_diff[metric])


            for (user_id, session_index), df_session in df_reaction_test.groupby(['user_id', 'session_index']):
                scores_pairwise_diff = {}
                for metric in (basic_metrics + ['rank_score', 'ensemble_baseline_score', 'AU4_valence_score']):
                    if metric == 'AU4_valence_score':
                        scores = - (1 - np.exp(-exp_decay_coef * df_session['AU4_activation_value'].to_numpy()))
                    else:
                        scores = df_session[metric].to_numpy()
                    scores_pairwise_diff[metric] = (scores[None,:] - scores[:,None]).flatten()

                compare_idx = np.where(np.logical_and(scores_pairwise_diff['rank_score'] != 0, np.abs(scores_pairwise_diff['AU4_valence_score']) >= diff_threshold))[0]
                for metric in (basic_metrics + ['rank_score', 'ensemble_baseline_score', 'AU4_valence_score']):
                    scores_pairwise_diff[metric] = scores_pairwise_diff[metric][compare_idx]
                    test_scores_pairwise_diff[metric].extend(scores_pairwise_diff[metric])
            
            for metric in (basic_metrics + ['ensemble_baseline_score', 'AU4_valence_score']):
                train_correctness = np.array(train_scores_pairwise_diff['rank_score']) * np.array(train_scores_pairwise_diff[metric]) > 0
                n_correct, n_incorrect = np.sum(train_correctness), np.sum(~train_correctness)
                train_n_correct[metric][AU4_sel_classification_coefs] = n_correct
                train_n_incorrect[metric][AU4_sel_classification_coefs] = n_incorrect
                
                test_correctness = np.array(test_scores_pairwise_diff['rank_score']) * np.array(test_scores_pairwise_diff[metric]) > 0
                n_correct, n_incorrect = np.sum(test_correctness), np.sum(~test_correctness)
                test_n_correct[metric][AU4_sel_classification_coefs] = n_correct
                test_n_incorrect[metric][AU4_sel_classification_coefs] = n_incorrect
                
        best_AU4_sel_classification_coefs = max(AU4_sel_classification_coefs_list, key=lambda coefs: train_n_correct['AU4_valence_score'][coefs] - 2*train_n_incorrect['AU4_valence_score'][coefs])
        best_exp_decay_coef, best_diff_threshold = best_AU4_sel_classification_coefs
        data_dict['exp_decay_coef'].append(best_exp_decay_coef)
        data_dict['diff_threshold'].append(best_diff_threshold)
        
        for metric in (basic_metrics + ['ensemble_baseline_score', 'AU4_valence_score']):
            data_dict[f"AU4_selected_{metric}_n_correct"].append(test_n_correct[metric][best_AU4_sel_classification_coefs])
            data_dict[f"AU4_selected_{metric}_n_incorrect"].append(test_n_incorrect[metric][best_AU4_sel_classification_coefs])
    
    
    df_reaction_all['AU4_valence_score'] = - (1 - np.exp(-np.vectorize(lambda u: data_dict['exp_decay_coef'][data_dict['user_id'].index(u)])(df_reaction_all['user_id']) * df_reaction_all['AU4_activation_value'].to_numpy()))
    
    
    # Compute the performance of all scores
    for user_id in filtered_user_id_list:
        df_participant = df_reaction_all[df_reaction_all['user_id'] == user_id].copy()

        participant_scores_pairwise_diff = defaultdict(list)
        for session in np.unique(df_participant['session_index']):
            df_session = df_participant[df_participant['session_index'] == session].copy()

            scores_pairwise_diff = {}
            for metric in (basic_metrics + ['rank_score', 'AU4_valence_score']):
                scores = df_session[metric].to_numpy()
                scores_pairwise_diff[metric] = (scores[None,:] - scores[:,None]).flatten()
            
            compare_idx = np.where(scores_pairwise_diff['rank_score'] != 0)[0]
            for metric in (basic_metrics + ['rank_score', 'AU4_valence_score']):
                scores_pairwise_diff[metric] = scores_pairwise_diff[metric][compare_idx]
                participant_scores_pairwise_diff[metric].extend(scores_pairwise_diff[metric])

        for metric in basic_metrics + ['AU4_valence_score']:
            correctness = np.array(participant_scores_pairwise_diff['rank_score']) * np.array(participant_scores_pairwise_diff[metric]) > 0
            n_correct, n_incorrect = np.sum(correctness), np.sum(~correctness)
            data_dict[f"{metric}_n_correct"].append(n_correct)
            data_dict[f"{metric}_n_incorrect"].append(n_incorrect)
    
    
    
    # Integrate AU4 valence score with the pre-trained scoring models
    AU4_coef_list = np.arange(0, 3.0+0.1, 0.1)

    for (baseline_score, ensemble_score) in [('ImageReward_score', 'ensemble_ImageReward_AU4_score'), ('PickScore', 'ensemble_PickScore_AU4_score'), ('HPS_V2_score', 'ensemble_HPS_V2_AU4_score'), ('ensemble_baseline_score', 'ensemble_all_score')]:
        for user_id_idx, left_out_user_id in enumerate(filtered_user_id_list):
            df_reaction_train = df_reaction_all[df_reaction_all['user_id'] != left_out_user_id].copy()
            df_reaction_test = df_reaction_all[df_reaction_all['user_id'] == left_out_user_id].copy()

            for metric in basic_metrics:
                train_mean, train_std = df_reaction_train[metric].mean(), df_reaction_train[metric].std()
                df_reaction_train[metric] = (df_reaction_train[metric] - train_mean) / train_std
                df_reaction_test[metric] = (df_reaction_test[metric] - train_mean) / train_std
                
            if baseline_score == 'ensemble_baseline_score':
                df_reaction_train['ensemble_baseline_score'] = np.vectorize(lambda u: data_dict['ImageReward_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['ImageReward_score'].to_numpy() + np.vectorize(lambda u: data_dict['PickScore_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['PickScore'].to_numpy() + np.vectorize(lambda u: data_dict['HPS_V2_weight'][data_dict['user_id'].index(u)])(df_reaction_train['user_id']) * df_reaction_train['HPS_V2_score'].to_numpy()

                df_reaction_test['ensemble_baseline_score'] = np.vectorize(lambda u: data_dict['ImageReward_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['ImageReward_score'].to_numpy() + np.vectorize(lambda u: data_dict['PickScore_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['PickScore'].to_numpy() + np.vectorize(lambda u: data_dict['HPS_V2_weight'][data_dict['user_id'].index(u)])(df_reaction_test['user_id']) * df_reaction_test['HPS_V2_score'].to_numpy()

            train_n_correct = {}
            train_n_incorrect = {}
            test_n_correct = {}
            test_n_incorrect = {}
            for AU4_coef in AU4_coef_list:
                train_rank_scores_pairwise_diff = []
                train_ensemble_scores_pairwise_diff = []
                test_rank_scores_pairwise_diff = []
                test_ensemble_scores_pairwise_diff = []

                for (user_id, session_index), df_session in df_reaction_train.groupby(['user_id', 'session_index']):
                    rank_scores = df_session['rank_score'].to_numpy()

                    ensemble_scores = df_session[baseline_score].to_numpy() + AU4_coef * df_session['AU4_valence_score'].to_numpy()

                    rank_scores_pairwise_diff = (rank_scores[None,:] - rank_scores[:,None]).flatten()
                    ensemble_scores_pairwise_diff = (ensemble_scores[None,:] - ensemble_scores[:,None]).flatten()

                    compare_idx = np.where(rank_scores_pairwise_diff != 0)[0]
                    rank_scores_pairwise_diff = rank_scores_pairwise_diff[compare_idx]
                    ensemble_scores_pairwise_diff = ensemble_scores_pairwise_diff[compare_idx]

                    train_rank_scores_pairwise_diff.extend(rank_scores_pairwise_diff)
                    train_ensemble_scores_pairwise_diff.extend(ensemble_scores_pairwise_diff)

                for (user_id, session_index), df_session in df_reaction_test.groupby(['user_id', 'session_index']):
                    rank_scores = df_session['rank_score'].to_numpy()

                    ensemble_scores = df_session[baseline_score].to_numpy() + AU4_coef * df_session['AU4_valence_score'].to_numpy()

                    rank_scores_pairwise_diff = (rank_scores[None,:] - rank_scores[:,None]).flatten()
                    ensemble_scores_pairwise_diff = (ensemble_scores[None,:] - ensemble_scores[:,None]).flatten()

                    compare_idx = np.where(rank_scores_pairwise_diff != 0)[0]
                    rank_scores_pairwise_diff = rank_scores_pairwise_diff[compare_idx]
                    ensemble_scores_pairwise_diff = ensemble_scores_pairwise_diff[compare_idx]

                    test_rank_scores_pairwise_diff.extend(rank_scores_pairwise_diff)
                    test_ensemble_scores_pairwise_diff.extend(ensemble_scores_pairwise_diff)

                train_correctness = np.array(train_rank_scores_pairwise_diff) * np.array(train_ensemble_scores_pairwise_diff) > 0
                n_correct, n_incorrect = np.sum(train_correctness), np.sum(~train_correctness)
                train_n_correct[AU4_coef] = n_correct
                train_n_incorrect[AU4_coef] = n_incorrect

                test_correctness = np.array(test_rank_scores_pairwise_diff) * np.array(test_ensemble_scores_pairwise_diff) > 0
                n_correct, n_incorrect = np.sum(test_correctness), np.sum(~test_correctness)
                test_n_correct[AU4_coef] = n_correct
                test_n_incorrect[AU4_coef] = n_incorrect

            best_AU4_coef = max(AU4_coef_list, key=train_n_correct.get)
            data_dict[f'AU4_coef_for_{ensemble_score}'].append(best_AU4_coef)

            data_dict[f'{ensemble_score}_n_correct'].append(test_n_correct[best_AU4_coef])
            data_dict[f'{ensemble_score}_n_incorrect'].append(test_n_incorrect[best_AU4_coef])
    
    
    
    df_results = pd.DataFrame(data_dict)
    os.makedirs(results_rootdir, exist_ok=True)
    df_results.to_csv(os.path.join(results_rootdir, 'image_preference_binary_classification_based_on_ranking.csv'), index=False)
    

    # Compute the heatmap of performance of selective classification based on AU4 valence score (for different coefficients)
    exp_decay_coef_list = np.arange(0+0.1, 2.1+0.1, 0.1)
    diff_threshold_list = np.arange(0.0, 1.0+0.02, 0.02)
    AU4_sel_classification_coefs_list = list(itertools.product(exp_decay_coef_list, diff_threshold_list))
    all_n_correct = {}
    all_n_incorrect = {}
    for AU4_sel_classification_coefs in AU4_sel_classification_coefs_list:
        exp_decay_coef, diff_threshold = AU4_sel_classification_coefs

        all_scores_pairwise_diff = defaultdict(list)

        for (user_id, session_index), df_session in df_reaction_all.groupby(['user_id', 'session_index']):
            scores_pairwise_diff = {}
            for metric in ['rank_score', 'AU4_valence_score']:
                if metric == 'AU4_valence_score':
                    scores = - (1 - np.exp(-exp_decay_coef * df_session['AU4_activation_value'].to_numpy()))
                else:
                    scores = df_session[metric].to_numpy()
                scores_pairwise_diff[metric] = (scores[None,:] - scores[:,None]).flatten()

            compare_idx = np.where(np.logical_and(scores_pairwise_diff['rank_score'] != 0, np.abs(scores_pairwise_diff['AU4_valence_score']) >= diff_threshold))[0]
            for metric in ['rank_score', 'AU4_valence_score']:
                scores_pairwise_diff[metric] = scores_pairwise_diff[metric][compare_idx]
                all_scores_pairwise_diff[metric].extend(scores_pairwise_diff[metric])

        correctness = np.array(all_scores_pairwise_diff['rank_score']) * np.array(all_scores_pairwise_diff['AU4_valence_score']) > 0
        n_correct, n_incorrect = np.sum(correctness), np.sum(~correctness)
        all_n_correct[AU4_sel_classification_coefs] = n_correct
        all_n_incorrect[AU4_sel_classification_coefs] = n_incorrect
    
    with open(os.path.join(results_rootdir, 'image_preference_binary_classification_based_on_ranking_AU4_correctness_heatmap.pkl'), 'wb') as f:
        pkl.dump({'exp_decay_coef_list': exp_decay_coef_list, 'diff_threshold_list': diff_threshold_list, 'n_correct': all_n_correct, 'n_incorrect': all_n_incorrect}, f)

if __name__ == '__main__':
    main()