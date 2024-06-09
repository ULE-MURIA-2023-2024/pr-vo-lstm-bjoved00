
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Callable


class VisualOdometryDataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        transform: Callable,
        sequence_length: int,
        validation: bool = False
    ) -> None:

        self.sequences = []

        directories = [d for d in os.listdir(
            dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        for subdir in directories:

            aux_path = f"{dataset_path}/{subdir}"

            # read data
            rgb_paths = self.read_images_paths(aux_path)

            
            ground_truth_data = self.read_ground_truth(aux_path)
            interpolated_ground_truth = self.interpolate_ground_truth(
                rgb_paths, ground_truth_data)

            # TODO: create sequences
            for i in range(1, len(rgb_paths), 2):
                
                position_first_image = np.array(interpolated_ground_truth[i-1][1])
                position_second_image = np.array(interpolated_ground_truth[i][1])
                difference = position_second_image - position_first_image

                self.sequences.append({"first_image_path": rgb_paths[i-1][1],
                                       "second_image_path": rgb_paths[i][1],
                                       "first_image_timestamp": rgb_paths[i-1][0],
                                       "second_image_timestamp": rgb_paths[i][0],
                                       "distance": difference})
                

        self.transform = transform
        self.sequence_length = sequence_length
        self.validation = validation

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.TensorType:

        # Load sequence of images
        sequence_images = []
        timestamp = 0
        distance = 0

        # TODO: return the next sequence
        secuencia = self.sequences[idx]

        timestamp = secuencia["second_image_timestamp"]

        img1 = cv2.imread(secuencia["first_image_path"])
        img2 = cv2.imread(secuencia["second_image_path"])

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        sequence_images = [img1, img2]
        sequence_images = torch.stack(sequence_images)

        distance = torch.Tensor(secuencia["distance"])

        return sequence_images, distance, timestamp

    def read_images_paths(self, dataset_path: str) -> Tuple[float, str]:

        paths = []

        with open(f"{dataset_path}/rgb.txt", "r") as file:
            for line in file:

                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                image_path = f"{dataset_path}/{line[1]}"

                paths.append((timestamp, image_path))

        return paths

    def read_ground_truth(self, dataset_path: str) -> Tuple[float, Tuple[float]]:

        ground_truth_data = []

        with open(f"{dataset_path}/groundtruth.txt", "r") as file:

            for line in file:

                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                position = list(map(float, line[1:]))
                ground_truth_data.append((timestamp, position))

        return ground_truth_data

    def interpolate_ground_truth(
            self,
            rgb_paths: Tuple[float, str],
            ground_truth_data: Tuple[float, Tuple[float]]
    ) -> Tuple[float, Tuple[float]]:

        rgb_timestamps = [rgb_path[0] for rgb_path in rgb_paths]
        ground_truth_timestamps = [item[0] for item in ground_truth_data]

        # Interpolate ground truth positions for each RGB image timestamp
        interpolated_ground_truth = []

        for rgb_timestamp in rgb_timestamps:

            nearest_idx = np.argmin(
                np.abs(np.array(ground_truth_timestamps) - rgb_timestamp))

            interpolated_position = ground_truth_data[nearest_idx]
            interpolated_ground_truth.append(interpolated_position)

        return interpolated_ground_truth
