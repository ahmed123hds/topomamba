import os
import numpy as np
import medmnist
from medmnist import INFO

def download_and_save_pathmnist(output_path, size=224):
    print(f"Downloading PathMNIST (size={size})...")
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # Download and load all splits
    train_dataset = DataClass(split='train', size=size, download=True)
    val_dataset = DataClass(split='val', size=size, download=True)
    test_dataset = DataClass(split='test', size=size, download=True)

    print("Saving to NPZ...")
    # The training script expects specific keys: train_images, train_labels, etc.
    np.savez(
        output_path,
        train_images=train_dataset.imgs,
        train_labels=train_dataset.labels,
        val_images=val_dataset.imgs,
        val_labels=val_dataset.labels,
        test_images=test_dataset.imgs,
        test_labels=test_dataset.labels
    )
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    out = os.path.join(os.getcwd(), "pathmnist_224.npz")
    download_and_save_pathmnist(out)
