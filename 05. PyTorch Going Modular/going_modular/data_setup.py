import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()
def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
):
    """Creates training and testing DataLoaders
    Args:
        train_dir: Path to the training directory
        test_dir: Path to the testing directory
        transform: torchvision transforms to perform on training and testing data
        batch_size: Number of samples per batch in each of the DataLoaders
        num_workers: An integer for number of workers per DataLoader

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
    """
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # get class names
    class_names = train_data.class_names

    # data loaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers = num_workers,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  num_workers = num_workers,
                                  shuffle=False)
    
    return train_dataloader, test_dataloader, class_names