import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.amp import GradScaler

scaler = GradScaler()

def train_super_resolution(model, dataloader, num_epochs=10, lr=1e-4, device='cuda'):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.to(device)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for i, (lr_img, hr_img, filename) in enumerate(dataloader):
            lr_img = lr_img.to(device, dtype=torch.float)
            hr_img = hr_img.to(device, dtype=torch.float)

            optimizer.zero_grad()

            with torch.autocast(device_type=device):
                sr_out = model(lr_img)
                loss = criterion(sr_out, hr_img)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if epoch==num_epochs-1:
                sr_outs = sr_out.cpu().detach()
                i =0
                for sr_out in sr_outs:
                    sr_out = sr_out.permute(1, 2, 0)  

                    import numpy as np
                    sr_out = np.clip(sr_out, 0, 1)
                    plt.imsave(f"results/{filename[0]}",sr_out, format='png')
                    i+=1

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        del loss, sr_out  
        torch.cuda.empty_cache()

        torch.save(model.state_dict(), "resnet_superres.pth")
        plot_loss(train_losses,"Training")

    plot_loss(train_losses,"Training")
    print("Training complete.")

def plot_loss(losses,loss_type, results_folder="results"):

    plt.figure(figsize=(8, 6))
    plt.plot(losses, label=f"{loss_type} Loss")
    plt.title(f"Epoch vs {loss_type} Loss")
    plt.xlabel("Epoch")
    plt.ylabel(f"{loss_type} Loss")
    plt.legend()

    plt.savefig(f'{results_folder}/{loss_type}_Loss.png')