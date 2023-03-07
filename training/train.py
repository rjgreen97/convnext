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

model = ConvNeXt().to(device)

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
)


def load_data():
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.data_root_dir, train=True, download=True, transform=transforms
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.data_root_dir, train=False, download=True, transform=transforms
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    return train_dataloader, test_dataloader


def train(train_dataloader):
    model.train()
    best_validation_accuracy = 0

    for epoch in range(1, config.num_epochs):
        loop = tqdm(train_dataloader)
        for data in loop:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch}/{config.num_epochs}")
            loop.set_postfix(loss=loss.item())

        print(f"Training Loss: {loss.item():.3f}")

        accuracy = validate(test_dataloader)
        if accuracy > best_validation_accuracy:
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_checkpoint_path)
            best_validation_accuracy = accuracy
        else:
            epochs_without_improvement += 1
            print(f"Epochs without improvement: {epochs_without_improvement}")
            if epochs_without_improvement > config.patience:
                print(f"Early stopping, patience of {config.patience} reached")
                break
    print("Finished Training")
    print(f"Best validation accuracy: {best_validation_accuracy:.2f}")


def validate(test_dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        accuracy = 100 * correct_predictions / total_predictions
    print(f"Validation Loss: {loss.item():.3f}")
    print(f"Validation Accuracy: {accuracy:.2f}")
    return accuracy


train_dataloader, test_dataloader = load_data()
train(train_dataloader)
