import os
import torch
from torch.utils.data import Dataset as TorchDataset
from imageio import imread
from torchvision import datasets, transforms
from tqdm import tqdm
import multiprocessing
from PIL import Image
from skimage.io import imread
from sklearn.model_selection import train_test_split

class Dataset(TorchDataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = imread(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)


        return image, label

def get_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []
    class_names = os.listdir(data_dir)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_to_idx[class_name])
    return image_paths, labels

def get_data_loaders(train_dir, test_dir, batch_size, transform=None):
    train_image_paths, train_labels = get_image_paths_and_labels(train_dir)
    test_image_paths, test_labels = get_image_paths_and_labels(test_dir)

    train_dataset = Dataset(train_image_paths, train_labels, transform)
    print(f"Train dataset: {train_dataset}")
    print(f"Train dataset size: {len(train_dataset)}")
    test_dataset = Dataset(test_image_paths, test_labels, transform)
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Compute mean and std of the dataset
def compute_mean_and_std(data_dir):
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """
    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)
        return d["mean"], d["std"]

    ds = datasets.ImageFolder(
        data_dir, transform=transforms.Compose([transforms.ToTensor()])
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=multiprocessing.cpu_count()
    )

    mean = torch.zeros(3)
    var = torch.zeros(3)
    npix = 0

    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean and std", ncols=80):
        images = images.view(3, -1)
        npix += images.size(1)
        mean += images.mean(1)
        var += images.var(1, unbiased=False)

    mean /= len(ds)
    std = torch.sqrt(var / len(ds))

    # Cache results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std