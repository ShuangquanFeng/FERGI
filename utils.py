import torch
from torchvision import transforms
import numpy as np
import pandas as pd
import pickle as pkl
import warnings
from config import *

def select_valid_frames(df): # Select valid frames of a video clip
    return (df['face_detection_confidence_score'] >= FDCS_min) & (df['pitch_indicative_ratio'].between(PIR_min, PIR_max)) & (df['yaw_indicative_ratio'].between(YIR_min, YIR_max))

def transforms_compose(steps, include_random=True): # Compose the transform steps
    if steps[0][0] == 'BGR2RGB':
        BGR2RGB = True
        steps = steps[1:]
    else:
        BGR2RGB = False       
    
    transform_list = []
    if BGR2RGB == True:
        transform_list.append(transforms.Lambda(lambda img: img[:,:,::-1]))
    transform_list.append(transforms.ToPILImage())
    for transform, inp in steps:
        if transform.startswith('Random') and include_random == False:
            continue
        if transform == 'RandomHorizontalFlip':
            transform_list.append(transforms.RandomHorizontalFlip())
        elif transform == 'RandomResizedCrop':
            transform_list.append(transforms.RandomResizedCrop(size=inp['size'], scale=inp['scale'], ratio=inp['ratio']))
        elif transform == 'Resize':
            transform_list.append(transforms.Resize((inp, inp)))
        else:
            raise ValueError('Unrecognized transformation.')
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)
        
def compute_frequency(all_labels, device): # Compute the frequency of each intensity of each AU in the dataset
    intensity_counts = torch.zeros((n_all_AUs, max_intensity + 1), dtype=torch.int, device=device)

    for labels in all_labels:
        for i in range(n_all_AUs):
            intensity = labels[i]
            if intensity >= 0:
                intensity_counts[i, int(intensity)] += 1
    
    threshold_counts = torch.zeros((n_all_AUs, max_intensity, 2), dtype=torch.int, device=device)

    for i in range(n_all_AUs):
        for j in range(1, max_intensity + 1):
            threshold_counts[i, j-1, 0] = torch.sum(intensity_counts[i, :j])
            threshold_counts[i, j-1, 1] = torch.sum(intensity_counts[i, j:])

    return intensity_counts, threshold_counts

def compute_weights(counts, method, AUs, device, groups = None): # Compute the weights of each intensity of each AU for the dataset
    counts_grouped = counts.clone()
    if groups != None: # Intensities in the same group should have the same weight
        for group in groups:
            counts_grouped[:,group] = torch.sum(counts[:,group], axis=1, keepdim=True, dtype=counts.dtype)
    weights = torch.zeros_like(counts_grouped, dtype=torch.float, device=device)
    if method == 'uniform':
        weights[counts_grouped > 0] = 1
    elif method == 'inverse':
        weights[counts_grouped > 0] = 1 / counts_grouped[counts_grouped > 0]
    elif method == 'inverse_sqrt':
        weights[counts_grouped > 0] = 1 / np.sqrt(counts_grouped[counts_grouped > 0])
    elif method == 'inverse_log':
        counts[counts_grouped > 0] += 1
        weights[counts_grouped > 0] = 1 / np.log(counts_grouped[counts_grouped > 0])
    else:
        raise ValueError("Unknown weight option. It should be 'uniform', 'inverse', 'inverse_sqrt', or 'inverse_log'.")
    final_weights = torch.zeros_like(weights, dtype=torch.float, device=device)
    final_weights[np.array(AUs)-1] = weights[np.array(AUs)-1] / weights[np.array(AUs)-1].sum(dim=tuple(range(1, final_weights.dim())), keepdim=True)
    return final_weights

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ICC(cse, typ, dat):
    """
    Adapted from Matlab function in https://github.com/ZhiwenShao/ARL
    Compute Intraclass Correlation Coefficients (ICC)

    Parameters:
        cse (int): 1, 2, or 3 depending on the raters configuration
        typ (str): 'single' or 'k' based on whether the ICC is based on a single measurement or on an average
        dat (array): data with raters/ratings in rows and targets in columns
    """

    # number of raters/ratings
    k = dat.shape[1]
    # number of targets
    n = dat.shape[0]
    # mean per target
    mpt = np.mean(dat, axis=1)
    # mean per rater/rating
    mpr = np.mean(dat, axis=0)
    # get total mean
    tm = np.mean(mpt)
    # within target sum squares
    WSS = np.sum((dat - mpt[:, np.newaxis])**2)
    # within target mean squares
    WMS = WSS / (n * (k - 1))
    # between rater sum squares
    RSS = np.sum((mpr - tm)**2) * n
    # between rater mean squares
    RMS = RSS / (k - 1)
    # between target sum squares
    BSS = np.sum((mpt - tm)**2) * k
    # between targets mean squares
    BMS = BSS / (n - 1)
    # residual sum of squares
    ESS = WSS - RSS
    # residual mean squares
    EMS = ESS / ((k - 1) * (n - 1))

    if cse == 1:
        if typ == 'single':
            return (BMS - WMS) / (BMS + (k - 1) * WMS)
        elif typ == 'k':
            return (BMS - WMS) / BMS
        else:
            raise ValueError('Wrong value for input typ')
    elif cse == 2:
        if typ == 'single':
            return (BMS - EMS) / (BMS + (k - 1) * EMS + k * (RMS - EMS) / n)
        elif typ == 'k':
            return (BMS - EMS) / (BMS + (RMS - EMS) / n)
        else:
            raise ValueError('Wrong value for input typ')
    elif cse == 3:
        if typ == 'single':
            return (BMS - EMS) / (BMS + (k - 1) * EMS)
        elif typ == 'k':
            return (BMS - EMS) / BMS
        else:
            raise ValueError('Wrong value for input typ')
    else:
        raise ValueError('Wrong value for input cse')

def compute_icc_3_1(pred_labels, true_labels): # Compute ICC(3,1)
    n_all_AUs = pred_labels.shape[1]

    icc_3_1_values = []

    for au in range(n_all_AUs):
        if np.all(true_labels[:, au] == -1):
            icc_3_1_values.append(float('nan'))
        else:
            icc_3_1_values.append(ICC(3, 'single', np.hstack((pred_labels[:,au:(au+1)], true_labels[:,au:(au+1)]))))
    return icc_3_1_values

def train_AU_model(model, train_dataset, train_loader, criterion, optimizer, scheduler, reg_weights_table, class_weights_table, loss_weights, device, write_results=False, results_path=None): # Train the AU model
    model.train()

    all_paths = []
    
    running_regression_loss_mse = 0.0
    running_regression_loss_cos = 0.0
    running_classification_loss = 0.0
    all_pred_labels_regression = []
    all_pred_labels_classification = []
    all_true_labels = []
    
    for paths, images, labels in train_loader:
        
        all_paths.extend(paths)
        
        this_batch_size = labels.shape[0]
        
        inputs = images.to(device)
        labels = labels.to(device)
        labels_classification = (labels.unsqueeze(-1) >= torch.arange(1, max_intensity + 1).to(device)).float()
        
        # Compute the weights in this batch based on the intensities
        i_indices, j_indices = torch.meshgrid(torch.arange(labels.shape[0], device=device), torch.arange(labels.shape[1], device=device))
        reg_weights = reg_weights_table[j_indices, labels.long()]

        i_indices, j_indices, k_indices = torch.meshgrid(
            torch.arange(labels_classification.shape[0], device=device), 
            torch.arange(labels_classification.shape[1], device=device), 
            torch.arange(labels_classification.shape[2], device=device)
        )
        class_weights = class_weights_table[j_indices, k_indices, labels_classification.long()]

        optimizer.zero_grad()

        outputs = model(inputs)
        
        # Compute the loss
        regression_loss_mse, regression_loss_cos, classification_loss = criterion(outputs, labels, reg_weights, class_weights)
        total_loss = loss_weights[0] * regression_loss_mse + loss_weights[1] * regression_loss_cos + loss_weights[2] * classification_loss
        total_loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()

        running_regression_loss_mse += (regression_loss_mse.item() * this_batch_size)
        running_regression_loss_cos += (regression_loss_cos.item() * this_batch_size)
        running_classification_loss += (classification_loss.item() * this_batch_size)

        outputs_regression = outputs[:, :n_all_AUs]
        pred_labels_regression = outputs_regression.data.cpu().numpy()
        outputs_classification = outputs[:,n_all_AUs:]
        pred_labels_classification = sigmoid(outputs_classification.data.cpu().numpy().reshape([this_batch_size, n_all_AUs, max_intensity]))
        pred_labels_classification = pred_labels_classification.sum(axis=-1)
        true_labels = labels.data.cpu().numpy()
        
        all_pred_labels_regression.append(pred_labels_regression)
        all_pred_labels_classification.append(pred_labels_classification)
        all_true_labels.append(true_labels)

    # Compute the overall loss, ICC, MSE, and MAE
    mse_loss = running_regression_loss_mse / len(train_dataset)
    cos_loss = running_regression_loss_cos / len(train_dataset)
    class_loss = running_classification_loss / len(train_dataset)

    pred_labels_regression = np.concatenate(all_pred_labels_regression)
    pred_labels_classification = np.concatenate(all_pred_labels_classification)
    true_labels = np.concatenate(all_true_labels)

    combination_weights = [0.5, 0.5]
    pred_labels_combination = combination_weights[0] * pred_labels_regression + combination_weights[1] * pred_labels_classification
    
    iccs_regression = compute_icc_3_1(pred_labels_regression, true_labels)
    iccs_classification = compute_icc_3_1(pred_labels_classification, true_labels)
    iccs_combination = compute_icc_3_1(pred_labels_combination, true_labels)
    
    mask = true_labels != -1
    masked_pred_labels_regression = np.where(mask, pred_labels_regression, np.nan)
    masked_pred_labels_classification = np.where(mask, pred_labels_classification, np.nan)
    masked_pred_labels_combination = np.where(mask, pred_labels_combination, np.nan)
    masked_true_labels = np.where(mask, true_labels, np.nan)

    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        
        mses_regression = np.nanmean((masked_pred_labels_regression - masked_true_labels) ** 2, axis=0)
        maes_regression = np.nanmean(np.abs(masked_pred_labels_regression - masked_true_labels), axis=0)

        mses_classification = np.nanmean((masked_pred_labels_classification - masked_true_labels) ** 2, axis=0)
        maes_classification = np.nanmean(np.abs(masked_pred_labels_classification - masked_true_labels), axis=0)

        mses_combination = np.nanmean((masked_pred_labels_combination - masked_true_labels) ** 2, axis=0)
        maes_combination = np.nanmean(np.abs(masked_pred_labels_combination - masked_true_labels), axis=0)

    print('Training Results:')
    print("Regression Loss (MSE): {:.3f}".format(mse_loss))
    print("Regression Loss (Cosine): {:.3f}".format(cos_loss))
    print("Classification Loss: {:.3f}".format(class_loss))
    print("ICCs Regression: [{}]".format(', '.join(f'{i:.3f}' for i in iccs_regression)))
    print("MSEs Regression: [{}]".format(', '.join(f'{i:.3f}' for i in mses_regression)))
    print("MAEs Regression: [{}]".format(', '.join(f'{i:.3f}' for i in maes_regression)))
    print("ICCs Classification: [{}]".format(', '.join(f'{i:.3f}' for i in iccs_classification)))
    print("MSEs Classification: [{}]".format(', '.join(f'{i:.3f}' for i in mses_classification)))
    print("MAEs Classification: [{}]".format(', '.join(f'{i:.3f}' for i in maes_classification)))
    print("ICCs Combination: [{}]".format(', '.join(f'{i:.3f}' for i in iccs_combination)))
    print("MSEs Combination: [{}]".format(', '.join(f'{i:.3f}' for i in mses_combination)))
    print("MAEs Combination: [{}]".format(', '.join(f'{i:.3f}' for i in maes_combination)))
    print('\n')

    # Save the results
    if write_results == True:
        results = {'image_paths': all_paths,
                   'pred_labels_regression': pred_labels_regression,
                   'pred_labels_classification': pred_labels_classification,
                   'true_labels': true_labels}
        with open(results_path, 'wb') as f:
            pkl.dump(results, f)

    return mse_loss, cos_loss, class_loss, iccs_regression, mses_regression, maes_regression, iccs_classification, mses_classification, maes_classification, iccs_combination, mses_combination, maes_combination