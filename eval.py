import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(folder1, folder2):
    """
    Computes the average PSNR and SSIM between images in two folders.

    """
    # Get list of image files (you can adjust the extensions if needed)
    valid_ext = ('.png', '.jpg', '.jpeg')
    files1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(valid_ext)])
    files2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(valid_ext)])
    
    # Match files by intersection of file names
    common_files = set(files1) & set(files2)
    if not common_files:
        raise ValueError("No common image files found between the two folders.")
    
    psnr_values = []
    ssim_values = []
    
    for filename in common_files:
        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)
        
        # Load images using OpenCV (BGR format)
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print(f"Warning: Could not load {filename} from one of the folders. Skipping.")
            continue
        
        # Ensure images have the same shape
        if img1.shape != img2.shape:
            print(f"Warning: Shape mismatch for {filename}: {img1.shape} vs {img2.shape}. Skipping.")
            continue
        
        # Option 1: Compute metrics on grayscale images.
        # Convert to grayscale:
        img1_gray = img1
        img2_gray = img2
        
        psnr = peak_signal_noise_ratio(img1_gray, img2_gray, data_range=255)
        ssim, _ = structural_similarity(img1_gray, img2_gray, full=True, data_range=255)
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
    
    avg_psnr = np.mean(psnr_values) if psnr_values else None
    avg_ssim = np.mean(ssim_values) if ssim_values else None
    
    return avg_psnr, avg_ssim

if __name__ == "__main__":
    # Update these folder paths as needed
    folders = r"D:\Users\r2shaji\613researchproject\results\fake"
    subfolders = [ f.name for f in os.scandir(folders) if f.is_dir() ]
    for folder in subfolders:

        folder2 = r"D:\Users\r2shaji\613researchproject\results\fake/" + folder
        folder1 = r"D:\Users\r2shaji\613researchproject\results\hr/" + folder
        avg_psnr, avg_ssim = compute_metrics(folder1, folder2)
        print("folder name",folder)
        print("Average PSNR:", avg_psnr)
        print("Average SSIM:", avg_ssim)
        print("\n\n")
