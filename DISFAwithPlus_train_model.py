import argparse
import torch
import torch.optim as optim
import pickle as pkl
import time
import os
import random
from config import *
from utils import *
from preprocess_datasets import *
from AU_datasets_preparation import *
from AU_dataset import *
from combination_loss import *
from iresnet_modified import *

# Sample bash command: 
# python DISFAwithPlus_train_model.py --epochs 3 --batch_size 64 --gpu 0 --save_interval 1


# The setup of model training
seed = 42

model_name = 'IR50'
if model_name == 'IR50':
    img_len = 112

model_pretrained_with = 'face recognition'
model_fine_tune = 'all'

model_attributes = {'name': model_name, 'pretrained': model_pretrained_with, 'fine-tune': model_fine_tune}

sel_AUs = DISFA_AUs.copy()

preprocess_steps = [('cvtColor', 'BGR2RGB'),
                    ('cropping', ('mediapipe', {'ymin': -0.2, 'ymax': 0.05, 'xmin': -0.125, 'xmax': 0.125})),
                    ('alignment', 112),
                    ('heNlm', (0.5, 0.5)),
                    ('padding', None),
                    ('cvtColor', 'RGB2BGR')]

transform_steps = [('BGR2RGB', None),
                   ('RandomHorizontalFlip', None),
                   ('RandomResizedCrop', {'size': img_len, 'scale': (0.90, 1), 'ratio': (19/20, 20/19)}),
                   ('Resize', img_len)]

dropout = 0
optimizer_settings = {'base_lr': 1e-5, 'last_layer_lr': 1e-4, 'weight_decay': 5e-4} 
class_weight_method = 'inverse'
class_weight_groups = [[0, 1], [2, 3, 4, 5]]
loss_weights = [1, 1, 1]

training_settings = {'dropout_rate': dropout, 'optimizer_settings': optimizer_settings, 'class_weight_method': class_weight_method, 'class_weight_groups': class_weight_groups, 'loss_weights': {'regression': loss_weights[0], 'cosine': loss_weights[1], 'classification': loss_weights[2]}}

all_setup = {'seed': seed, 'model': model_attributes, 'AUs': sel_AUs, 'preprocess': preprocess_steps, 'transform': transform_steps, 'training': training_settings}

parser = argparse.ArgumentParser(description='train AU model')
parser.add_argument('--epochs', default=3, type=int, help="number of epochs for training")
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--gpu', default=0, type=int, help="the GPU to use")
parser.add_argument('--save_interval', default=1, type=int, help='define the interval of epochs to save model state')

def main():
    args = parser.parse_args()
    if seed is not None:
        # Make the model training deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    num_epochs = args.epochs
    batch_size = args.batch_size
    interval = args.save_interval
    
    data_transforms = {}
    data_transforms['train'] = transforms_compose(transform_steps, include_random=True)
    
    results_folder = 'AU_models'
    os.makedirs(results_folder, exist_ok=True)
    setup_path = os.path.join(results_folder, 'training_setup.pkl')
    with open(setup_path, 'wb') as f:
        pkl.dump(all_setup, f)
        
    # Preprocess the dataset
    print("Preprocessing the dataset...")
    
    device = torch.device('cuda:' + str(args.gpu))
    
    preprocess_steps_str = preprocess_datasets(DISFA_img_rootdir_leftcam, preprocess_steps, device)
    preprocess_steps_str = preprocess_datasets(DISFA_img_rootdir_rightcam, preprocess_steps, device)
    preprocess_steps_str = preprocess_datasets(DISFAPlus_img_rootdir, preprocess_steps, device)

    # Initialize the model
    print("Initializing the model...")
    if model_name == 'IR50':
        if model_pretrained_with == 'none':
            model = iresnet_basic(n_layers=50, n_features = n_all_AUs * (max_intensity + 1), dropout=dropout)
        elif model_pretrained_with == 'face recognition':
            model = iresnet_basic(n_layers=50, n_features = n_all_AUs * (max_intensity + 1), pretrained = True, weights_path = 'pretrained_models/glint360k_cosface_r50_fp16_0.1.pth', dropout=dropout)
            if model_fine_tune == 'all':
                pass
            elif model_fine_tune == 'last stage':
                for name, child in model.named_children():
                    if name != 'layer4':
                        for param in child.parameters():
                            param.requires_grad = False
                    else:
                        break
    else:
        ValueError('Unknown model option.')

    model = model.to(device)

    torch.cuda.device(args.gpu)

    # Initialize the dataset and dataloader
    print("Initializing Dataset and Dataloader...")
    train_dataset = AU_dataset(subject_dict = {'DISFAleft': DISFA_subjects, 'DISFAright': DISFA_subjects, 'DISFAPlus': DISFAPlus_subjects}, preprocess_steps_str = preprocess_steps_str, transform = data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Model training settings
    last_layer = list(model.children())[-1]
    try:
        last_layer = last_layer[-1]
    except:
        last_layer = last_layer
    ignored_params = list(map(id, last_layer.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params and p.requires_grad, model.parameters())
    optimizer = optim.Adam([{'params': base_params, 'lr': optimizer_settings['base_lr']},
                            {'params': last_layer.parameters(), 'lr': optimizer_settings['last_layer_lr']}], weight_decay = optimizer_settings['weight_decay'])

    criterion = CombinationLoss(n_all_AUs, max_intensity, reduction='mean')
    scheduler = None

    # Compute the weights of different labels
    print("Computing weights...")

    intensity_counts, threshold_counts = compute_frequency(train_dataset.all_labels, device)

    reg_weights_table = compute_weights(intensity_counts, class_weight_method, sel_AUs, device, groups=class_weight_groups)
    class_weights_table = compute_weights(threshold_counts, class_weight_method, sel_AUs, device)

    metrics_path = os.path.join(results_folder, 'metrics.pkl')
    metrics = []

    # Train the model
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        start_time = time.time() 
        
        if (epoch+1) % interval == 0:
            write_results = True
            train_results_path = os.path.join(results_folder, f"training_results_epoch{epoch+1}.pkl")
        else:
            write_results = False
            train_results_path = None

        train_mse_loss, train_cos_loss, train_class_loss, train_iccs_regression, train_mses_regression, train_maes_regression, train_iccs_classification, train_mses_classification, train_maes_classification, train_iccs_combination, train_mses_combination, train_maes_combination = train_AU_model(model, train_dataset, train_loader, criterion, optimizer, scheduler, reg_weights_table, class_weights_table, loss_weights, device, write_results=write_results, results_path=train_results_path)
        
        # Save results
        epoch_metrics = {
            'epoch': epoch+1,
            'train': {
                'mse_loss': train_mse_loss,
                'cos_loss': train_cos_loss,
                'class_loss': train_class_loss,
                'iccs_regression': train_iccs_regression,
                'mses_regression': train_mses_regression,
                'maes_regression': train_maes_regression,
                'iccs_classification': train_iccs_classification,
                'mses_classification': train_mses_classification,
                'maes_classification': train_maes_classification,
                'iccs_combination': train_iccs_combination,
                'mses_combination': train_mses_combination,
                'maes_combination': train_maes_combination
            }
        }
        metrics.append(epoch_metrics)
        with open(metrics_path, 'wb') as f:
            pkl.dump(metrics, f)

        if (epoch+1) % interval == 0:
            torch.save(model.state_dict(), os.path.join(results_folder, 'checkpoint_epoch'+str(epoch+1)+'.pth'))

        end_time = time.time()
        time_elapsed = end_time - start_time
        print('Epoch {} complete in {:.0f}m {:.0f}s\n'.format(epoch+1, time_elapsed // 60, time_elapsed % 60))



if __name__ == '__main__':
    main()