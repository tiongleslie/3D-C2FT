import torch.utils.data.dataset
from torchvision import transforms
import os
import scipy
import numpy as np
from typing import Tuple, List
import random
import json
import cv2
from datetime import datetime
from PIL import Image
from tqdm import tqdm

from data import binvox_rw
from data import data_transforms


class ShapeNet_Dataset(torch.utils.data.Dataset):

    def __init__(self, config, dataset_type, train_augmentation=False, preload_2d_to_ram=False, preload_3d_to_ram=False):
        """
        Args:
        :param config: configuration settings for ShapeNet
        :param dataset_type: train, valid or test
        :param train_augmentation: use augmentation
        :param preload_2d_to_ram: load 2D images into ram, default=False
        :param preload_3d_to_ram: load 3D voxels into ram, default=False
        """
        self.taxonomy_path = config.ShapeNet_taxonomy_path
        self.image_path = config.ShapeNet_image_path
        self.voxel_path = config.ShapeNet_voxel_path

        self.dataset_type = dataset_type

        self.n_views = config.n_views
        self.selection_mode = config.ShapeNet_selection_mode
        self.class_dict = config.class_dict
        self.train_augmentation = train_augmentation

        if train_augmentation:
            self.transforms = data_transforms.Compose([
                data_transforms.RandomCrop(config.img_size, config.crop_size),
                data_transforms.RandomBackground(config.train_random_bg_color_range),
                data_transforms.ColorJitter(config.brightness, config.contrast, config.saturation),
                data_transforms.RandomNoise(config.noise_std),
                data_transforms.Normalize(mean=config.mean, std=config.std),
                data_transforms.RandomFlip(),
                data_transforms.RandomPermuteRGB(),
                data_transforms.ToTensor(),
            ])
        else:
            self.transforms = data_transforms.Compose([
                data_transforms.CenterCrop(config.img_size, config.crop_size),
                data_transforms.RandomBackground(config.eval_random_bg_color_range),
                data_transforms.Normalize(mean=config.mean, std=config.std),
                data_transforms.ToTensor(),
            ])

        self.check_options()

        self.file_list = self.build_file_list()
        self.real_size = len(self.file_list)

        self.preload_2d_to_ram = preload_2d_to_ram
        self.preload_3d_to_ram = preload_3d_to_ram
        if self.preload_2d_to_ram:
            self.images_2d = []
            self.preload_2d_data()
        if self.preload_3d_to_ram:
            self.objects_3d = []
            self.preload_3d_data()

    def preload_2d_data(self):
        print("Preloading 2d dataset into RAM")
        for i, file_path in enumerate(tqdm(self.file_list)):
            self.images_2d.append(np.asarray([self.read_img(image_path) for image_path in file_path['image_paths']]))

    def preload_3d_data(self):
        print("Preloading 3d dataset into RAM")
        for i, file_path in enumerate(tqdm(self.file_list)):
            self.objects_3d.append(self.read_volume(file_path['volume']))

    def select_images_from_preload(self, idx):
        selected_images = [self.images_2d[idx][i] for i in range(self.n_views)]
        return selected_images

    def check_options(self) -> None:
        """
            Validate given arguments
        """
        assert self.dataset_type in ['train', 'val', 'test']
        assert isinstance(self.n_views, int) and 1 <= self.n_views <= 24
        assert self.selection_mode in ['random', 'fixed']

    def __len__(self) -> int:
        """
            Return the dataset size.
        :return: Dataset size
        """
        return self.real_size

    def __getitem__(self, idx: int):
        """
            Get a single dataset item
        Args:
        :param idx: Item position in the file_list
        :return: A set of images, GT volume and sample info
        """
        idx = idx % self.real_size
        taxonomy_name, sample_name, images, volume = self.get_datum(idx)

        if self.transforms:
            # Apply pre-processing to the images
            images = self.transforms(images)

        if self.preload_2d_to_ram and self.selection_mode == 'random':
            idx = torch.randperm(images.shape[0])
            images = images[idx].view(images.size())

        return images, volume, self.class_dict[taxonomy_name]

    def get_datum(self, idx: int) -> Tuple[str, str, List, np.ndarray]:
        """
            Load a single data sample
        Args:
        :param idx: Position of the data sample in the file list
        :return: Sample info, images and GT volume
        """
        # Grab the sample info and paths
        sample = self.file_list[idx]
        taxonomy_name = sample['taxonomy_name']  # string, like '02691156'
        sample_name = sample['sample_name']  # string, like '6c432109eee42aed3b053f623496d8f5'

        if self.preload_2d_to_ram:
            # load from preloaded array
            images = self.select_images_from_preload(idx)
        else:
            # Select images to use
            selected_image_paths = self.select_images(sample['image_paths'], self.n_views)
            # Read RGB Images
            images = [self.read_img(image_path) for image_path in selected_image_paths]
            images = np.asarray(images)

        if self.preload_3d_to_ram:
            volume = self.objects_3d[idx]  # volume.shape => [N_VOX (32), N_VOX (32), N_VOX (32)]
        else:
            # Read 3D GT Volume
            volume = self.read_volume(sample['volume'])  # volume.shape => [N_VOX (32), N_VOX (32), N_VOX (32)]

        return taxonomy_name, sample_name, images, volume

    def build_file_list(self):
        """
            Builds a list of all files
        :return: List of all files
        """
        # Grab image and volume paths
        taxonomy_path = self.taxonomy_path
        image_path_template = self.image_path
        volume_path_template = self.voxel_path

        # Load all taxonomies of the dataset
        with open(taxonomy_path, encoding='utf-8') as file:
            dataset_taxonomy = json.loads(file.read())

        files = []

        # Load data for each category
        for taxonomy in dataset_taxonomy:
            # Command-line update
            self.log_taxonomy_loading(taxonomy)

            # Get a list of files for a single category
            taxonomy_id = taxonomy['taxonomy_id']
            samples = taxonomy[self.dataset_type]
            taxonomy_files = self.get_files_of_taxonomy(taxonomy_id, samples,
                                                        image_path_template, volume_path_template)

            # Combine category-level file list to the overall list
            files.extend(taxonomy_files)

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (datetime.now(), len(files)))
        return files

    @staticmethod
    def get_files_of_taxonomy(taxonomy_folder_id, samples, image_path_template, volume_path_template):
        """
            Builds the file list of a single category
        Args:
        :param taxonomy_folder_id: Category ID
        :param samples: List of sample IDs
        :param image_path_template: Template path to the images folder
        :param volume_path_template: Template path to the volume folder
        :return: List of files of a single category
        """
        TOTAL_VIEWS = 24
        all_image_indices = range(TOTAL_VIEWS)
        files_of_taxonomy = []

        for sample_idx, sample_name in enumerate(samples):
            # List of all images for a single sample
            image_paths = [
                image_path_template % (taxonomy_folder_id, sample_name, image_idx)
                for image_idx in all_image_indices
            ]

            # Path to the ground-truth volume
            volume_file_path = volume_path_template % (taxonomy_folder_id, sample_name)

            # Append to the list of files
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_id,
                'sample_name': sample_name,
                'image_paths': image_paths,
                'volume': volume_file_path
            })
        return files_of_taxonomy

    def select_images(self, all_image_paths: List[str], num_images: int) -> List[str]:
        """
            Select images to use for a single sample
        Args:
        :param all_image_paths: Path to all images
        :param num_images: Number of images to choose
        :return: List of selected image paths
        """
        if self.selection_mode == 'random':
            selected_ids = random.sample(range(len(all_image_paths)), num_images)
            selected_image_paths = [all_image_paths[i] for i in selected_ids]
        else:
            selected_image_paths = [all_image_paths[i] for i in range(num_images)]
        return selected_image_paths

    @staticmethod
    def read_img(path: str) -> np.ndarray:
        """
            Read an image at the given path
        Args:
        :param path: Path to the image
        :return: Loaded image, shape: [H (137), W (137), C (4)]
        """
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # Ensure that the image is either RGB or RGBA
        assert len(image.shape) in [3, 4], \
            print('[FATAL] %s It seems that there is something wrong with the image file %s' % (datetime.now(), path))
        return image

    @staticmethod
    def read_volume(path):
        """
            Read 3D volume at the given path
        Args:
        :param path: Path to the volume
        :return: Loaded 3D volume, shape: [N_VOX (32), N_VOX (32), N_VOX (32)]
        """
        _, extension = os.path.splitext(path)

        if extension == '.mat':
            volume = scipy.io.loadmat(path)
            volume = volume['Volume'].astype(np.float32)
        elif extension == '.binvox':
            with open(path, 'rb') as f:
                volume = binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)
        return volume

    def log_taxonomy_loading(self, taxonomy):
        """
            Update command-line with taxonomy loading message
        :param taxonomy: Taxonomy info
        """
        sample_count = len(taxonomy[self.dataset_type])
        print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s] %d' %
              (datetime.now(), taxonomy['taxonomy_id'], taxonomy['taxonomy_name'], sample_count))
