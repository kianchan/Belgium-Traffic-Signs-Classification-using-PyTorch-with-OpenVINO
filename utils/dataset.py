import os
import zipfile
import requests
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

TRAIN_URL = "http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip"
TEST_URL = "http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip"


def download_and_extract(url, extract_to="data"):
    os.makedirs(extract_to, exist_ok=True)
    filename = os.path.join(extract_to, url.split("/")[-1])

    # Skip download if zip exists
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        with requests.get(url, stream=True) as r:
            total = int(r.headers.get('content-length', 0))
            with open(filename, 'wb') as f, tqdm(
                desc=filename, total=total, unit='B', unit_scale=True
            ) as bar:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
                    bar.update(len(chunk))

    # Extract only if training/testing folders not already present
    folder_name = os.path.splitext(os.path.basename(filename))[0]
    extracted_dir = os.path.join(extract_to, folder_name.replace("BelgiumTSC_", ""))  # Training / Testing
    if not os.path.exists(extracted_dir):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)


class BelgiumDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        self.classes = sorted(os.listdir(root))

        valid_exts = (".jpg", ".jpeg", ".png", ".ppm", ".bmp")
        for class_id, cls in enumerate(self.classes):
            cls_path = os.path.join(root, cls)
            if os.path.isdir(cls_path):
                for img_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img_name)
                    if not img_name.lower().endswith(valid_exts):
                        continue
                    self.samples.append((img_path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(transform, batch_size=32):
    # Check if Training and Testing already exist before extracting
    if not (os.path.exists("data/Training") and os.path.exists("data/Testing")):
        download_and_extract(TRAIN_URL, "data")
        download_and_extract(TEST_URL, "data")

    train_dir = "data/Training"
    test_dir = "data/Testing"

    train_dataset = BelgiumDataset(train_dir, transform)
    test_dataset = BelgiumDataset(test_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, len(train_dataset.classes)
