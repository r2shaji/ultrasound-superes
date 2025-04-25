from prefect import flow, task
import torch, os
from torch.utils.data import DataLoader

from networks import UNet
from dataset import UltrasoundDataset
from train import train_super_resolution

@task
def load_dataset(root_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = UltrasoundDataset(root_dir=root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

@task
def init_model(n_channels: int, n_channels_out: int, bilinear: bool, device: str, model_path: str) -> torch.nn.Module:
    model = UNet(n_channels=n_channels, n_channels_out=n_channels_out, bilinear=bilinear)
    model = model.to(device)
    if model_path is not None and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model state dict from {model_path}")
    else:
        print("No model found at the specified path. Initializing a new model.")
    return model

@task
def train_model(model: torch.nn.Module, dataloader: DataLoader, num_epochs: int, lr: float, device: str) -> torch.nn.Module:
    train_super_resolution(model, dataloader, num_epochs=num_epochs, lr=lr, device=device)
    return model

@task
def save_model(model: torch.nn.Module, file_path: str) -> None:
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

@flow(name="SuperResolution Training Flow")
def main_flow(
    root_dir: str = r"D:\Users\r2shaji\Downloads\PICMUS Images with Windows\none Window Images",
    batch_size: int = 4,
    num_epochs: int = 100,
    lr: float = 1e-4,
    num_workers: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_path: str = "resnet_superres.pth"
):
    dataloader = load_dataset(root_dir, batch_size, num_workers)
    model = init_model(n_channels=1, n_channels_out=1, bilinear=False, device=device, model_path=model_path)
    trained_model = train_model(model, dataloader, num_epochs, lr, device)
    save_model(trained_model, "resnet_superres.pth")

if __name__ == "__main__":
    # You can pass parameters here if desired
    main_flow()
