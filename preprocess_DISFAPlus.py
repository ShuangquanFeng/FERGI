import argparse
import torch
from config import *
from preprocess_datasets import *

parser = argparse.ArgumentParser(description='preprocess dataset')
parser.add_argument('--gpu', default=0, type=int, help="the GPU to use")

def main():
    args = parser.parse_args()
    device = torch.device('cuda:' + str(args.gpu))
    
    # Preprocess the DISFA+ dataset
    preprocess_steps = [('cvtColor', 'BGR2RGB'),
                        ('cropping', ('mediapipe', {'ymin': -0.2, 'ymax': 0.05, 'xmin': -0.125, 'xmax': 0.125})),
                        ('alignment', 112),
                        ('heNlm', (0.5, 0.5)),
                        ('padding', None),
                        ('cvtColor', 'RGB2BGR')]

    preprocess_steps_str = preprocess_datasets(DISFAPlus_img_rootdir, preprocess_steps, device)
    
if __name__ == '__main__':
    main()