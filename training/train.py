import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models.convnext import ConvNeXt, InvertedBottleneck
from training.training_config import TrainingConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)

config = TrainingConfig()

model_checkpoint_path = config.model_checkpoint_path

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.49139968, 0.48215841, 0.44653091),
            std=(0.24703223, 0.24348513, 0.26158784),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(2, 5),  # unsupported by PIL
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ]
)

model = ConvNeXt().to(device)
