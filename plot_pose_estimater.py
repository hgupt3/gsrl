import argparse
import logging
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataloaders import pose_dataset
from models.networks import pose_estimation_net, SITR_base
import seaborn as sns
import matplotlib.pyplot as plt
import os

def get_args():
    """
    Parse command line arguments for model configuration and evaluation parameters.
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', '-m', default='SITR_base', help='From "SITR_base"')
    parser.add_argument('--load', '-f', type=str, default='checkpoints/pose_estimation/', help='Load model from a .pth file')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--calibration-config', dest='cc', default=18, type=int, help='From 0, 4, 8, 9, 18')
    parser.add_argument('--device', '-d', default='cuda:0', help='Device to train on')
    parser.add_argument('--val-path', type=str, default='pose_dataset/val_set', help='Validation set path')
    return parser.parse_args()

def evaluate(net, dataset, args):
    """
    Evaluate the pose estimation model's performance on a given dataset.
    
    Args:
        net: Neural network model
        dataset: Dataset to evaluate on
        args: Command line arguments containing evaluation parameters
    
    Returns:
        float: Average RMSE loss across the dataset
    """
    device = torch.device(args.device)
    net.eval()
    net.to(device)
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    loss = nn.MSELoss()
    val_loss = 0

    # Iterate over the validation set
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Validation round', unit='batch', leave=False):
            # Extract initial and final samples, labels, and calibration data
            samples_i, samples_f, labels, calibs = batch['sample_init'], batch['sample_final'], batch['label'], batch['calibration']
            
            # Move data to appropriate device
            samples_i = samples_i.to(device=device)
            samples_f = samples_f.to(device=device)
            labels = labels.to(device=device)
            calibs = calibs.to(device=device)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                # Forward pass through the network
                pred = net(samples_i, samples_f, calibs)
                loss_val = loss(pred, labels)
                val_loss += loss_val.item()
            
    # Calculate average RMSE across all batches
    avg_rmse = torch.sqrt(torch.tensor(val_loss / len(test_dataloader)))
    print(f"Average RMSE: {avg_rmse:.4f}")
    return avg_rmse.item()

def confusion_across_sensor(list_of_sensor, net, args, output_file):
    """
    Create and save a confusion matrix showing RMSE loss across different sensors.
    
    Args:
        list_of_sensor: List of sensor IDs to evaluate
        net: Neural network model
        args: Command line arguments
        output_file: Path to save the confusion matrix plot
    """
    # Define the classes to evaluate
    class_list = [0,2,3,4,5,7,8,9,10,11,13,14,15,16,17,18]
    conf_matrix = np.zeros((len(list_of_sensor), len(list_of_sensor)))
    
    # Get the device to use
    device = torch.device(args.device)
    
    # Evaluate model performance across all sensor combinations
    for i, sensor in enumerate(list_of_sensor):
        # Load weights with proper device mapping
        weights_path = os.path.join(args.load, args.base_model, f'sensor_000{sensor}.pth')
        weights = torch.load(weights_path, map_location=device)
        net.load_state_dict(weights)
        net.to(device)  # Ensure model is on correct device
        
        for j, sensor2 in enumerate(list_of_sensor):
            print(f'Evaluating: Sensor {sensor} → Sensor {sensor2}')
            # Create dataset for current sensor combination
            dataset = pose_dataset(
                path=args.val_path,
                sensor_list=[sensor2],
                calibration_config=args.cc,
                augment=False,
                random_final=False
            )
            conf_matrix[i][j] = evaluate(net, dataset, args)
            
    # Create and save the confusion matrix plot
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='.2f',
        cmap='viridis_r',  # Reversed viridis colormap
        square=True,
        cbar_kws={'label': 'RMSE'},
        xticklabels=list_of_sensor,
        yticklabels=list_of_sensor,
        vmin=0.5, 
        vmax=2.5  
    )

    ax.set_xlabel('Tested on sensor #')
    ax.set_ylabel('Trained on sensor #')
    ax.set_title('RMSE across sensors')
    plt.savefig(output_file)
    plt.close()  # Close the figure to free memory

if __name__ == '__main__':    
    args = get_args()
    
    # Initialize the appropriate model based on command line arguments
    if args.base_model == 'SITR_base':
        net = pose_estimation_net(SITR_base(num_calibration=args.cc))
    
    # Generate confusion matrix for the first group of sensors
    confusion_across_sensor([0,1,2,3], net, args, output_file='pos_confusion_matrix_inter.png')