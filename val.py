
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import VisualOdometryDataset
from model import VisualOdometryModel
from params import *


# Create the visual odometry model
model = VisualOdometryModel(hidden_size, 
                            num_layers, 
                            bidirectional, 
                            lstm_dropout)

transform = T.Compose([
    T.ToTensor(),
    model.resnet_transforms()
])


# TODO: Load dataset
dataset = VisualOdometryDataset(dataset_path="./dataset/val", 
                                      transform=transform, 
                                      sequence_length=sequence_length,
                                      validation=True)

valid_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4)

# val
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
model.load_state_dict(torch.load("vo.pt"))
model.eval()

validation_string = ""
position = [0.0] * 7

with torch.no_grad():
    for images, labels, timestamp in tqdm(valid_loader, f"Validating:"):

        images = images.to(device)
        labels = labels.to(device)

        target = model(images).cpu().numpy().tolist()[0]

        # TODO: add the results into the validation_string
        position = [pos + delta for pos, delta in zip(position, target)]
        position_str = ' '.join(map(str, position))
        validation_string += f"{timestamp[0]} {position_str}\n"


f = open("validation.txt", "a")
f.write(validation_string)
f.close()
