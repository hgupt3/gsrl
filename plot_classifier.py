import argparse
import logging
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataloaders import classification_dataset
from models.networks import classification_net, SITR_base
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os

def get_args():
    """
    Parse command line arguments for model configuration and training parameters.
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', '-m', default='SITR_base', help='From "SITR_base"')
    parser.add_argument('--load', '-f', type=str, default='checkpoints/classification/', help='Load model from a folder')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--calibration-config', dest='cc', default=18, type=int, help='From 0, 4, 8, 9, 18')
    parser.add_argument('--device', '-d', default='cuda:0', help='Device to train on')
    parser.add_argument('--val-path', type=str, default='classification_dataset/val_set', help='Validation set path')
    return parser.parse_args()

def evaluate(net, dataset, args):
    """
    Evaluate the model's performance on a given dataset.
    
    Args:
        net: Neural network model
        dataset: Dataset to evaluate on
        args: Command line arguments containing evaluation parameters
    
    Returns:
        float: Accuracy percentage
    """
    device = torch.device(args.device)
    net.eval()
    net.to(device)
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    correct = 0
    total = 0

    # Iterate over the validation set
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Validation round', unit='batch', leave=False):
            samples, calibs, labels = batch['sample'], batch['calibration'], batch['label']

            samples = samples.to(device=device)
            calibs = calibs.to(device=device)
            labels = labels.to(device=device)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                class_pred = net(samples, calibs)  
                _, predicted = torch.max(class_pred.data, 1)
                _, labels = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def confusion_across_sensor(list_of_sensor, net, args, output_file):
    """
    Create and save a confusion matrix showing accuracy across different sensors.
    
    Args:
        list_of_sensor: List of sensor IDs to evaluate
        net: Neural network model
        args: Command line arguments
        output_file: Path to save the confusion matrix plot
    """
    # Define the classes to evaluate
    class_list = [0,2,3,4,5,7,8,9,10,11,13,14,15,16,17,18]
    conf_matrix = np.zeros((len(list_of_sensor), len(list_of_sensor)))
    
    # Evaluate model performance across all sensor combinations
    for i, sensor in enumerate(list_of_sensor):
        weights = torch.load(os.path.join(args.load, args.base_model, f'sensor_000{sensor}.pth'))
        net.load_state_dict(weights)
        for j, sensor2 in enumerate(list_of_sensor):
            print(f'Evaluating: Sensor {sensor} → Sensor {sensor2}')
            dataset = classification_dataset(
                path=args.val_path,
                sensor_list=[sensor2],
                class_list=class_list,
                augment=False,
                calibration_config=args.cc
            )
            conf_matrix[i][j] = evaluate(net, dataset, args)
            
    # Create and save the confusion matrix plot
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        square=True,
        cbar_kws={'label': 'Accuracy'},
        xticklabels=list_of_sensor,
        yticklabels=list_of_sensor,
        vmin=0,
        vmax=100
    )

    ax.set_xlabel('Tested on sensor #')
    ax.set_ylabel('Trained on sensor #')
    ax.set_title('Accuracy across sensors')
    plt.savefig(output_file)
    plt.close()  # Close the figure to free memory

if __name__ == '__main__':
    args = get_args()
    
    # Initialize the appropriate model based on command line arguments
    if args.base_model == 'SITR_base':
        net = classification_net(SITR_base(num_calibration=args.cc))
    
    # Generate confusion matrices for different sensor groups
    confusion_across_sensor([0,1,2,3], net, args, output_file='class_confusion_matrix_inter.png')
    confusion_across_sensor([0,4,5,6], net, args, output_file='class_confusion_matrix_intra.png')