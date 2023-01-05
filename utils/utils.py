import torch
import random
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='/opt/ml/meca/runs')
    parser.add_argument('--data_dir', type=str, default='/opt/ml/meca/dataset/train_image')

    parser.add_argument('--use_model', type=str, default='vgg')  ## efficientnet, mobilenet, vgg
    parser.add_argument('--use_loss', type=str, default='cross_entropy')
    parser.add_argument('--use_optimizer', type=str, default='adam')
    parser.add_argument('--use_scheduler', type=str, default='cosine_annealing')

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)

    parser.add_argument('--experiment_name', '-en', type=str, default='vgg11')
    parser.add_argument('--seed', type=int, default=21)

    args = parser.parse_args()

    return args

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)