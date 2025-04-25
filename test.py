import torch.optim as optim
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_super_resolution(model, dataloader, device='cuda'):
    """
    Sets the model to evaluation mode, runs inference on each batch in the dataloader,
    and saves the super-resolved output images as well as the corresponding HR images.
    The SR images are saved in the "results" folder and the HR images in "results/hr".
    """
    # Ensure the output directories exist
    results_folder = "results"
    hr_folder = os.path.join(results_folder, "hr")
    fake_folder = os.path.join(results_folder, "fake")
    os.makedirs(fake_folder, exist_ok=True)
    os.makedirs(hr_folder, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i, (lr_img, hr_img, lr_path) in enumerate(dataloader):
            
            lr_img = lr_img.to(device, dtype=torch.float)
            with torch.autocast(device_type=device):
                sr_out = model(lr_img)
            sr_out = sr_out.cpu().detach()
            hr_img = hr_img.cpu().detach()

            for idx in range(sr_out.size(0)):
                print("lr_path[idx]",lr_path[idx])
                filename = os.path.basename(lr_path[idx])
                parent_dir = os.path.dirname(lr_path[idx])
                folder_name = os.path.basename(os.path.dirname(parent_dir))
                
                # permute from (C, H, W) to (H, W, C) and convert to numpy
                print("size", sr_out[idx].shape)
                sr_img = sr_out[idx].permute(1, 2, 0).numpy()
                if sr_img.shape[-1] == 1:
                    sr_img = np.repeat(sr_img, 3, axis=2)
                print("size", sr_img.shape)
                sr_img = np.clip(sr_img, 0, 1)
                print("size", sr_img.shape)
                sr_folder = os.path.join(fake_folder, folder_name)
                os.makedirs(sr_folder, exist_ok=True)
                sr_filename = os.path.join(sr_folder, f"{filename}")
                plt.imsave(sr_filename, sr_img, format='png')

                hr_img_item = hr_img[idx].permute(1, 2, 0).numpy()
                if hr_img_item.shape[-1] == 1:
                    hr_img_item = np.repeat(hr_img_item, 3, axis=2)
                hr_img_item = np.clip(hr_img_item, 0, 1)
                hr_folder_i = os.path.join(hr_folder, f"{folder_name}")
                os.makedirs(hr_folder_i, exist_ok=True)
                hr_filename = os.path.join(hr_folder_i, f"{filename}")
                plt.imsave(hr_filename, hr_img_item, format='png')

if __name__ == "__main__":
    import torch
    from networks import UNet
    from torch.utils.data import DataLoader
    from dataset import UltrasoundDataset

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    num_epochs = 10
    lr_value = 1e-4

    dataset = UltrasoundDataset(
        root_dir=r"D:\Users\r2shaji\Downloads\PICMUS Images with Windows\none Window Images"
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = UNet(n_channels=1, n_channels_out=1, bilinear=False)
    model = model.to(device)

    model.load_state_dict(torch.load("resnet_superres.pth", map_location=device))
    test_super_resolution(model, dataloader, device=device)