import os
import os.path as osp
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import random
from tqdm import tqdm
import albumentations as A

# Dataset statistics for normalization
# These values are pre-computed for the dataset with background subtraction
sample_mu = [-1.2223, -1.8114, -1.7090]
sample_std = [11.7932, 12.7956, 13.6452]

# Statistics for depth maps
dmap_mu = [8.8114]
dmap_std = [30.3624]

# Statistics for surface normals
norm_mu = [126.9855, 127.2061, 247.7740]
norm_std = [25.1953, 25.5532, 22.1101]

class sim_dataset(Dataset):
    """
    Custom dataset class for loading and processing sensor data.
    Handles calibration images, samples, depth maps, and surface normals.
    """
    def __init__(self, 
                 path, 
                 augment=False,
                 sendTwo=False, 
                 transforms=T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)]),
                 dmap_transforms=T.Compose([T.ToTensor(), T.Normalize(mean=dmap_mu, std=dmap_std)]),
                 norm_transforms=T.Compose([T.ToTensor(), T.Normalize(mean=norm_mu, std=norm_std)]),
                 calibration_config=18,
                 num_samples=None,
                 num_sensors=None) -> None:
        # Initialize dataset parameters and paths
        self.path = path
        self.transforms = transforms
        self.dmap_transforms = dmap_transforms
        self.norm_transforms = norm_transforms
        self.sendTwo = sendTwo

        # Configure calibration list based on calibration_config
        if calibration_config == 0: self.calib_list = []
        elif calibration_config == 4: self.calib_list = [1,3,7,9]
        elif calibration_config == 8: self.calib_list = [1,3,7,9,10,12,16,18]
        elif calibration_config == 9: self.calib_list = [i for i in range(1, 10)]
        elif calibration_config == 18: self.calib_list = [i for i in range(1, 19)]
        else: raise ValueError('Invalid calibration configuration')
            
        # Load and sort sensor directories
        sensors = os.listdir(path)
        sensors.sort()
        if '.DS_Store' in sensors: sensors.remove('.DS_Store')
        
        # Get calibration and sample information
        calibrations = os.listdir(osp.join(path, sensors[0], 'calibration'))
        if '.DS_Store' in calibrations: calibrations.remove('.DS_Store')
        
        samples =  os.listdir(osp.join(path, sensors[-1], 'samples'))
        if '.DS_Store' in samples: samples.remove('.DS_Store')
        
        # Set dataset size parameters
        self.num_calibrations = len(calibrations)
        
        if num_samples != None: self.num_samples = num_samples
        else: self.num_samples = len(samples)
        
        if num_sensors != None: self.num_sensors = num_sensors
        else: self.num_sensors = len(sensors)
        
        # Initialize augmentation pipeline
        self.augment = A.Compose([], additional_targets={'c0':'image', 
                                                         'c1':'image', 'c2':'image', 'c3':'image', 
                                                         'c4':'image', 'c5':'image', 'c6':'image', 
                                                         'c7':'image', 'c8':'image', 'c9':'image', 
                                                         'c10':'image', 'c11':'image', 'c12':'image', 
                                                         'c13':'image', 'c14':'image', 'c15':'image', 
                                                         'c16':'image', 'c17':'image', 'c18':'image'})
        if augment == True:
            # Define augmentation transforms for both sample and calibration images
            self.augment = A.Compose([A.ColorJitter(brightness=(0.6, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                                      A.Blur()],
                                      additional_targets={'c0':'image', 
                                                          'c1':'image', 'c2':'image', 'c3':'image', 
                                                          'c4':'image', 'c5':'image', 'c6':'image', 
                                                          'c7':'image', 'c8':'image', 'c9':'image', 
                                                          'c10':'image', 'c11':'image', 'c12':'image', 
                                                          'c13':'image', 'c14':'image', 'c15':'image', 
                                                          'c16':'image', 'c17':'image', 'c18':'image'})

    def __len__(self) -> int:
        return self.num_sensors * self.num_samples
    
    def getitem_helper(self, sensor_idx, sample_idx):
        """
        Helper function to load and process data for a specific sensor and sample.
        Handles background subtraction, calibration data loading, and augmentation.
        """
        # Load background reference image for subtraction
        ref_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'calibration', '0000.png')
        ref_img = np.array(Image.open(ref_path))
        
        # Load all calibration images
        calib = []
        for i in range(1,19):
            calib_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'calibration', '{0:04}.png'.format(i))
            calib_img = np.array(Image.open(calib_path))
            calib.append(calib_img)
        
        # Load sample image
        sample_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'samples', '{0:04}.png'.format(sample_idx))
        sample = np.array(Image.open(sample_path)) 
        
        # Apply augmentations to all images
        augments = self.augment(image=sample,
                                c0=ref_img,
                                c1=calib[0], c2=calib[1], c3=calib[2],
                                c4=calib[3], c5=calib[4], c6=calib[5], 
                                c7=calib[6], c8=calib[7], c9=calib[8],
                                c10=calib[9], c11=calib[10], c12=calib[11],
                                c13=calib[12], c14=calib[13], c15=calib[14],
                                c16=calib[15], c17=calib[16], c18=calib[17])
        
        # Process reference image
        ref_img = np.array(augments['c0'], dtype=np.float32)
        
        # Process calibration images with background subtraction
        calib = torch.tensor([])
        for i in self.calib_list:
            calib_img = np.array(augments[f'c{i}'], dtype=np.float32)
            calib_img = augments[f'c{i}'] - ref_img
            calib = torch.cat([calib, self.transforms(calib_img)])
            
        # Process sample image with background subtraction
        sample =  np.array(augments['image'], dtype=np.float32)
        sample = augments['image'] - ref_img
        sample = self.transforms(sample) 
        
        # Load depth map and surface normals if available
        try:
            dmap_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'dmaps', '{0:04}.png'.format(sample_idx))
            dmap = np.array(Image.open(dmap_path), dtype=np.float32)
            dmap = self.dmap_transforms(dmap)

            norm_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'norms', '{0:04}.png'.format(sample_idx))
            norm = np.array(Image.open(norm_path), dtype=np.float32)
            norm = self.norm_transforms(norm)
        except: 
            dmap = None
            norm = None
        
        idx = torch.tensor(sample_idx)
        
        return sample, calib, dmap, norm, idx
    
    def __getitem__(self, index):
        """
        Main data loading function that handles both single and paired data loading.
        """
        if index >= len(self): raise IndexError(f"Index {index} out of range")
        
        # Calculate sensor and sample indices
        sensor_idx = index // self.num_samples
        sample_idx = index % self.num_samples
        
        # Load primary data
        sample, calib, dmap, norm, idx = self.getitem_helper(sensor_idx, sample_idx)
        
        # Handle paired data loading for contrastive learning
        if self.sendTwo: 
            second_sensor_idx = sensor_idx
            while second_sensor_idx == sensor_idx:
                second_sensor_idx = random.randint(0,self.num_sensors-1)
            sample2, calib2, dmap2, norm2, _ = self.getitem_helper(second_sensor_idx, sample_idx)
            
            # Stack paired data
            calib = torch.stack([calib, calib2], dim=0)
            sample = torch.stack([sample, sample2], dim=0) 
            dmap = torch.stack([dmap, dmap2], dim=0)
            norm = torch.stack([norm, norm2], dim=0)
        
        return {'sample': sample, 'calibration': calib, 'dmap': dmap, 'norm' : norm, 'idx': idx}
    
class classification_dataset(Dataset):
    """
    Custom dataset class for classification tasks.
    Handles loading and preprocessing of sensor data with calibration images.
    """
    def __init__(self, 
                 path, 
                 augment=False,
                 transforms=T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)]),
                 calibration_config=18,
                 sensor_list = [],
                 class_list = [],
                 num_samples = None,
                 ) -> None:
        self.path = path
        self.transforms = transforms
        self.sensor_list = sensor_list
        
        # Configure calibration list based on calibration_config parameter
        if calibration_config == 0: self.calib_list = []
        elif calibration_config == 4: self.calib_list = [1,3,7,9]
        elif calibration_config == 8: self.calib_list = [1,3,7,9,10,12,16,18]
        elif calibration_config == 9: self.calib_list = [i for i in range(1, 10)]
        elif calibration_config == 18: self.calib_list = [i for i in range(1, 19)]
        else: raise ValueError('Invalid calibration configuration')
        
        # Clean up directory listings by removing .DS_Store files
        sensors = os.listdir(path)
        if '.DS_Store' in sensors: sensors.remove('.DS_Store')
        
        calibrations = os.listdir(osp.join(path, sensors[0], 'calibration'))
        if '.DS_Store' in calibrations: calibrations.remove('.DS_Store')

        classes = os.listdir(osp.join(path, sensors[0], 'samples'))
        if '.DS_Store' in classes: classes.remove('.DS_Store')
        
        samples =  os.listdir(osp.join(path, sensors[0], 'samples', classes[0]))
        if '.DS_Store' in samples: samples.remove('.DS_Store')
        
        # Set dataset dimensions
        self.num_sensors = len(sensor_list)
        self.num_calibrations = len(calibrations)
        self.num_classes = len(classes)
        
        if num_samples is not None: self.num_samples = num_samples
        else: self.num_samples = len(samples)
        
        if class_list:
            self.num_classes = len(class_list)
        self.class_list = class_list
        
        # Initialize augmentation pipeline
        self.augment = A.Compose([], additional_targets={'c0':'image', 
                                                         'c1':'image', 'c2':'image', 'c3':'image', 
                                                         'c4':'image', 'c5':'image', 'c6':'image', 
                                                         'c7':'image', 'c8':'image', 'c9':'image', 
                                                         'c10':'image', 'c11':'image', 'c12':'image', 
                                                         'c13':'image', 'c14':'image', 'c15':'image', 
                                                         'c16':'image', 'c17':'image', 'c18':'image'})
        if augment == True:
            # Configure data augmentation transforms
            self.augment = A.Compose([A.ColorJitter(brightness=(0.6, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                                      A.Blur()],
                                      additional_targets={'c0':'image', 
                                                          'c1':'image', 'c2':'image', 'c3':'image', 
                                                          'c4':'image', 'c5':'image', 'c6':'image', 
                                                          'c7':'image', 'c8':'image', 'c9':'image', 
                                                          'c10':'image', 'c11':'image', 'c12':'image', 
                                                          'c13':'image', 'c14':'image', 'c15':'image', 
                                                          'c16':'image', 'c17':'image', 'c18':'image'})

    def __len__(self) -> int:
        """Return total number of samples in the dataset"""
        return self.num_samples * self.num_classes * self.num_sensors
    
    def getitem_helper(self, sensor_idx, class_idx, sample_idx):    
        """
        Helper function to load and preprocess a single sample
        Returns processed sample, calibration data, and label
        """
        # Load background reference image for subtraction
        ref_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'calibration', '0000.png')
        ref_img = np.array(Image.open(ref_path))
        
        # Load all calibration images
        calib = []
        for i in range(1,19):
            calib_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'calibration', '{0:04}.png'.format(i))
            calib_img = np.array(Image.open(calib_path))
            calib.append(calib_img)
        
        # Load the actual sample image
        sample_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'samples', 'class_{0:04}'.format(class_idx), '{0:04}.png'.format(sample_idx))
        sample = np.array(Image.open(sample_path)) 
        
        # Apply augmentations to all images (sample and calibration)
        augments = self.augment(image=sample,
                                c0=ref_img,
                                c1=calib[0], c2=calib[1], c3=calib[2],
                                c4=calib[3], c5=calib[4], c6=calib[5], 
                                c7=calib[6], c8=calib[7], c9=calib[8],
                                c10=calib[9], c11=calib[10], c12=calib[11],
                                c13=calib[12], c14=calib[13], c15=calib[14],
                                c16=calib[15], c17=calib[16], c18=calib[17])
        
        # Process reference image
        ref_img = np.array(augments['c0'], dtype=np.float32)
        
        # Process calibration images: subtract background and apply transforms
        calib = torch.tensor([])
        for i in self.calib_list:
            calib_img = np.array(augments[f'c{i}'], dtype=np.float32)
            calib_img = augments[f'c{i}'] - ref_img
            calib = torch.cat([calib, self.transforms(calib_img)])
            
        # Process sample image: subtract background and apply transforms
        sample =  np.array(augments['image'], dtype=np.float32)
        sample = augments['image'] - ref_img
        sample = self.transforms(sample) 
        
        # Create one-hot encoded label
        if self.class_list:
            label = torch.nn.functional.one_hot(torch.tensor(self.class_list.index(class_idx)), num_classes=self.num_classes)
        else: label = torch.nn.functional.one_hot(torch.tensor(class_idx), num_classes=self.num_classes)
        label = label.float()
            
        return sample, calib, label
    
    def __getitem__(self, index):
        """
        Main method to get a sample from the dataset
        Returns a dictionary containing the sample, calibration data, and label
        """
        if index >= len(self): raise IndexError(f"Index {index} out of range")
        
        # Calculate indices for sensor, class, and sample
        sensor_idx = self.sensor_list[index // (self.num_samples * self.num_classes)]
        class_idx = (index % (self.num_samples * self.num_classes)) // self.num_samples
        if self.class_list:
            class_idx = self.class_list[class_idx]
        sample_idx = index % self.num_samples
        
        # Get the processed data
        sample, calib, label = self.getitem_helper(sensor_idx, class_idx, sample_idx)
        
        return {'sample': sample, 'calibration': calib, 'label': label}

class pose_dataset(Dataset):
    """
    Custom dataset class for pose estimation that handles sensor data with background subtraction.
    Supports data augmentation and multiple sensor inputs.
    """
    def __init__(self, 
                 path, 
                 augment=False,
                 transforms=T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)]),
                 sensor_list = [],
                 calibration_config = 18,
                 random_final = False,
                 ) -> None:
        self.path = path
        self.transforms = transforms
        self.sensor_list = sensor_list
        self.random_final = random_final
        
        if calibration_config == 0: self.calib_list = []
        elif calibration_config == 4: self.calib_list = [1,3,7,9]
        elif calibration_config == 8: self.calib_list = [1,3,7,9,10,12,16,18]
        elif calibration_config == 9: self.calib_list = [i for i in range(1, 10)]
        elif calibration_config == 18: self.calib_list = [i for i in range(1, 19)]
        else: raise ValueError('Invalid calibration configuration')
        
        # Get list of sensors, calibrations, and samples from the data directory
        sensors = os.listdir(path)
        if '.DS_Store' in sensors: sensors.remove('.DS_Store')
        
        calibrations = os.listdir(osp.join(path, sensors[0], 'calibration'))
        if '.DS_Store' in calibrations: calibrations.remove('.DS_Store')
        
        classes = os.listdir(osp.join(path, sensors[0], 'samples'))
        if '.DS_Store' in classes: classes.remove('.DS_Store')
        
        samples =  os.listdir(osp.join(path, sensors[0], 'samples', classes[0]))
        if '.DS_Store' in samples: samples.remove('.DS_Store')
        
        # Store counts for dataset size calculation
        self.num_sensors = len(sensor_list)
        self.num_calibrations = len(calibrations)
        self.num_classes = len(classes)
        self.num_samples = len(samples)
        
        self.augment = A.Compose([], additional_targets={'c0':'image', 
                                                         'c1':'image', 'c2':'image', 'c3':'image', 
                                                         'c4':'image', 'c5':'image', 'c6':'image', 
                                                         'c7':'image', 'c8':'image', 'c9':'image', 
                                                         'c10':'image', 'c11':'image', 'c12':'image', 
                                                         'c13':'image', 'c14':'image', 'c15':'image', 
                                                         'c16':'image', 'c17':'image', 'c18':'image'})
        if augment == True:
            # we need batched augmentations for calibration
            self.augment = A.Compose([A.ColorJitter(brightness=(0.6, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                                      A.Blur()],
                                      additional_targets={'c0':'image', 
                                                          'c1':'image', 'c2':'image', 'c3':'image', 
                                                          'c4':'image', 'c5':'image', 'c6':'image', 
                                                          'c7':'image', 'c8':'image', 'c9':'image', 
                                                          'c10':'image', 'c11':'image', 'c12':'image', 
                                                          'c13':'image', 'c14':'image', 'c15':'image', 
                                                          'c16':'image', 'c17':'image', 'c18':'image'})

    def __len__(self) -> int:
        return self.num_samples * self.num_classes * self.num_sensors
    
    def getitem_helper(self, sensor_idx, class_idx, sample_idx):    
        """
        Helper method to load and preprocess a single sample from the dataset.
        Handles background subtraction and data augmentation.
        """
        # Load background reference image for subtraction
        ref_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'calibration', '0000.png')
        ref_img = np.array(Image.open(ref_path))
        
        # Load calibration images
        calib = []
        for i in range(1,19):
            calib_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'calibration', '{0:04}.png'.format(i))
            calib_img = np.array(Image.open(calib_path))
            calib.append(calib_img)
            
        # Load sample image
        sample_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'samples', 'obj_{0:04}'.format(class_idx), '{0:04}.png'.format(sample_idx))
        sample = np.array(Image.open(sample_path)) 
        
        # Apply augmentations to all images
        augments = self.augment(image=sample,
                                c0=ref_img,
                                c1=calib[0], c2=calib[1], c3=calib[2],
                                c4=calib[3], c5=calib[4], c6=calib[5], 
                                c7=calib[6], c8=calib[7], c9=calib[8],
                                c10=calib[9], c11=calib[10], c12=calib[11],
                                c13=calib[12], c14=calib[13], c15=calib[14],
                                c16=calib[15], c17=calib[16], c18=calib[17])
        
        # Process reference image
        ref_img = np.array(augments['c0'], dtype=np.float32)
        
        # Process calibration images with background subtraction and normalization
        calib = torch.tensor([])
        for i in self.calib_list:
            calib_img = np.array(augments[f'c{i}'], dtype=np.float32)
            calib_img = augments[f'c{i}'] - ref_img
            calib = torch.cat([calib, self.transforms(calib_img)])
            
        # Process sample image with background subtraction and normalization
        sample =  np.array(augments['image'], dtype=np.float32)
        sample = augments['image'] - ref_img
        sample = self.transforms(sample) 
        
        # Load location data
        location_path = osp.join(self.path, 'sensor_{0:04}'.format(sensor_idx), 'locations', 'obj_{0:04}'.format(class_idx), '{0:04}.npy'.format(sample_idx))
        location = np.load(location_path)[:3]
        location = torch.tensor(location, dtype=torch.float32)
        
        return sample, calib, location
    
    def __getitem__(self, index):
        """
        Returns a data sample containing initial and final states, calibration data, and location change.
        """
        if index >= len(self): raise IndexError(f"Index {index} out of range")
        
        sensor_idx = self.sensor_list[index // (self.num_samples * self.num_classes)]
        class_idx = (index % (self.num_samples * self.num_classes)) // self.num_samples
        sample_idx = index % self.num_samples
        
        # Get initial state data
        sample_init, calib, location_init = self.getitem_helper(sensor_idx, class_idx, sample_idx)
        
        # Get final state data (either next sequential sample or random sample)
        if self.random_final: next_index = random.randint(0,self.num_samples-1)
        else: next_index = (index+1)%self.num_samples
        
        sample_final, _, location_final = self.getitem_helper(sensor_idx, class_idx, next_index)
        
        # Calculate location change between initial and final states
        location = location_final - location_init
        
        return {'sample_init': sample_init, 'sample_final': sample_final, 'calibration': calib, 'label': location, 'sensor': sensor_idx, 'idx':sample_idx, 'class':class_idx}
  
# Script to calculate dataset statistics (mean and std) for normalization
if __name__ == '__main__':
    # Initialize dataset without normalization for statistics calculation
    ds = sim_dataset(
        transforms=T.Compose([T.ToTensor()]), 
        dmap_transforms=T.Compose([T.ToTensor()]),
        norm_transforms=T.Compose([T.ToTensor()]),
    )
    
    # Initialize accumulators for statistics calculation
    rgb_sum = torch.tensor([0.0, 0.0, 0.0])
    rgb_sum_sq = torch.tensor([0.0, 0.0, 0.0])

    dmap_sum = torch.tensor([0.0])
    dmap_sum_sq = torch.tensor([0.0])

    norm_sum = torch.tensor([0.0, 0.0, 0.0])
    norm_sum_sq = torch.tensor([0.0, 0.0, 0.0])

    # Calculate running sums for mean and variance
    pbar = tqdm(ds)    
    for i, batch in enumerate(pbar):
        sample = batch['sample']
        dmap = batch['dmap']
        norm = batch['norm']
        
        rgb_sum += sample.sum(axis=[1,2])
        rgb_sum_sq += (sample**2).sum(axis=[1,2])

        dmap_sum += dmap.sum()
        dmap_sum_sq += (dmap**2).sum()

        norm_sum += norm.sum(axis=[1,2])
        norm_sum_sq += (norm**2).sum(axis=[1,2])

    # Calculate final statistics
    count = len(ds) * 224 * 224 

    rgb_mean = rgb_sum / count
    rgb_var = (rgb_sum_sq / count) - (rgb_mean**2)
    rgb_std = torch.sqrt(rgb_var)

    dmap_mean = dmap_sum / count
    dmap_var = (dmap_sum_sq / count) - (dmap_mean**2)
    dmap_std = torch.sqrt(dmap_var)

    norm_mean = norm_sum / count
    norm_var = (norm_sum_sq / count) - (norm_mean**2)
    norm_std = torch.sqrt(norm_var)
    
    # Print calculated statistics
    print("rgb mean: " + str(rgb_mean))
    print("rgb std:  " + str(rgb_std))
    print('')
    print("dmap mean: " + str(dmap_mean))
    print("dmap std:  " + str(dmap_std))
    print('')
    print("norm mean: " + str(norm_mean))
    print("norm std:  " + str(norm_std))
    