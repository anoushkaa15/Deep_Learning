# --- CELL 1: RUN THIS FIRST (ONLY ONCE PER RESTART) ---
import os
import sys
import subprocess
import time

print("--- Starting Full Setup ---")

# 1. Clone the repository
if not os.path.exists('UAV_Anomaly_Detection'):
    print("[1/5] Cloning repository...")
    subprocess.run(['git', 'clone', 'https://github.com/Geospatial-Computer-Vision-Group/UAV_Anomaly_Detection.git'], check=True, capture_output=True)
else:
    print("[1/5] Repository already cloned.")

# 2. Change directory
print("[2/5] Changing directory to Swin-MAE...")
try:
    %cd UAV_Anomaly_Detection/Swin-MAE
except Exception as e:
    print(f"       (Already in correct directory)")

# Verify we are in the right place
if not os.path.exists('swin_mae.py'):
    print("---!!!---")
    print("CRITICAL ERROR: Not in the correct directory. Restart your notebook and run this cell again.")
    print("---!!!---")
    sys.exit()
else:
    print("       Successfully in /UAV_Anomaly_Detection/Swin-MAE")

# 3. Patch requirements.txt
print("[3/5] Patching requirements.txt to fix version errors...")
try:
    # This comments out the old torch, torchvision, and numpy lines
    subprocess.run(["sed", "-i", "s/^torch==.*/# &/", "requirements.txt"], check=True)
    subprocess.run(["sed", "-i", "s/^torchvision==.*/# &/", "requirements.txt"], check=True)
    subprocess.run(["sed", "-i", "s/^numpy==.*/# &/", "requirements.txt"], check=True)
    print("       Successfully patched requirements.txt.")
except Exception as e:
    print(f"       Error patching requirements.txt: {e}")

# 4. Install requirements
print("[4/5] Installing requirements...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '--quiet'], check=True)
print("       Successfully installed requirements.")

# 5. Apply Python code patches (np.int and loss_weight_init)
print("[5/5] Patching swin_mae.py for np.int and loss_weight_init bugs...")
file_to_fix = 'swin_mae.py'
try:
    # Read the file
    with open(file_to_fix, 'r') as f:
        content = f.read()
    
    # Apply Patch 1 (Fix np.int error)
    if 'np.int' in content:
        content = content.replace('np.int', 'int')
        print("       Applied Patch: 'np.int' -> 'int'")
    else:
        print("       Patch 'np.int' was already applied.")
        
    # Apply Patch 2 (Fix loss_weight_init error)
    find_line = "super()._init_()"
    insert_line = "\n        self.loss_weight_init = 1.0"
    if insert_line not in content:
        content = content.replace(find_line, find_line + insert_line, 1)
        print("       Applied Patch: Added 'self.loss_weight_init = 1.0'")
    else:
        print("       Patch 'loss_weight_init' was already applied.")

    # Write the fixed content back
    with open(file_to_fix, 'w') as f:
        f.write(content)
    print("       Successfully patched swin_mae.py.")

except Exception as e:
    print(f"       An error occurred while trying to patch the file: {e}")

# Short delay to ensure file system is updated
time.sleep(2)



# --- Add this block to the END of Cell 1 ---
print("\n[6/5] Force-reinstalling Pillow & Torchvision to fix environment conflicts...")
try:
    # Uninstall both
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'Pillow', 'torchvision'], check=True, capture_output=True)
    # Reinstall both together to get compatible versions
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'Pillow', 'torchvision'], check=True, capture_output=True)
    print("       Successfully re-installed Pillow & Torchvision.")
except Exception as e:
    print(f"       Error re-installing packages: {e}")



# --- CELL 2: RUN THIS TO START ANOMALY PRE-TRAINING (DYNAMIC MSE STAGNATION) ---
import os
import sys
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import json

# --- MANUAL KNEE POINT FUNCTION (No external library needed) ---
def find_knee_point_manual(sorted_values):
    """
    Finds the knee point of a curve using the geometric method:
    Maximum vertical distance from the diagonal line.
    """
    n_points = len(sorted_values)
    if n_points < 2: return sorted_values[0]
    
    x = np.arange(n_points)
    y = sorted_values
    
    x_norm = (x - x[0]) / (x[-1] - x[0])
    y_min, y_max = y[0], y[-1]
    
    if y_max == y_min: return y_max
        
    y_norm = (y - y_min) / (y_max - y_min)
    difference_curve = x_norm - y_norm
    knee_idx = np.argmax(difference_curve)
    
    return y[knee_idx]

# --- 1. Imports from Cloned Repo ---
try:
    import swin_mae 
    import utils.misc as misc
    from utils.misc import NativeScalerWithGradNormCount as NativeScaler
    print("Successfully imported patched 'swin_mae' model.")
except ImportError:
    print("Error importing swin_mae. Please run Setup cell.")
    sys.exit()

try:
    import timm
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "timm"])
    import timm


# --- 2. Configuration ---
DATASET_ROOT = "/kaggle/input/agriculture-vision-dl/Agriculture-Vision-2021"
MODEL_SAVE_PATH = "swinmae_small_model.pth"
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 50 
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAE_MASK_RATIO = 0.75


# --- 3. Dataset & Mask Loader ---
def load_anomaly_masks(anomaly_labels_dir, filename, img_size=IMG_SIZE):
    ANOMALY_CLASSES = [
        'double_plant', 'drydown', 'endrow', 'nutrient_deficiency',
        'planter_skip', 'storm_damage', 'water', 'waterway', 'weed_cluster'
    ]
    base_name = os.path.splitext(os.path.basename(filename))[0] + ".png"
    combined_mask = None
    found_any = False

    def pil_to_mask_tensor(pil_img):
        pil_resized = pil_img.resize((img_size, img_size), resample=Image.NEAREST)
        tensor = transforms.ToTensor()(pil_resized)
        return (tensor > 0.5).float()

    for cls in ANOMALY_CLASSES:
        mask_path = os.path.join(anomaly_labels_dir, cls, base_name)
        if os.path.exists(mask_path):
            found_any = True
            try:
                pil_mask = Image.open(mask_path).convert("L")
            except Exception:
                continue
            mask_tensor = pil_to_mask_tensor(pil_mask)
            combined_mask = mask_tensor if combined_mask is None else torch.max(combined_mask, mask_tensor)

    if not found_any: return None
    return combined_mask


class AgriVision4ChDataset(Dataset):
    def init(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.img_rgb_dir = os.path.join(self.root_dir, "images", "rgb")
        self.img_nir_dir = os.path.join(self.root_dir, "images", "nir")
        self.anomaly_labels_dir = os.path.join(self.root_dir, "labels")
        self.image_ids = sorted(os.listdir(self.img_rgb_dir))
        self.transform = transform

    def len(self):
        return len(self.image_ids)

    def getitem(self, idx):
        filename = self.image_ids[idx]
        rgb_path = os.path.join(self.img_rgb_dir, filename)
        nir_path = os.path.join(self.img_nir_dir, filename)

        try:
            rgb_img = Image.open(rgb_path).convert("RGB")
            nir_img = Image.open(nir_path).convert("L")
        except Exception:
            dummy_img = torch.zeros((4, IMG_SIZE, IMG_SIZE))
            dummy_mask = torch.zeros((1, IMG_SIZE, IMG_SIZE))
            return dummy_img, dummy_mask

        if self.transform:
            rgb_img = self.transform(rgb_img)
            nir_img = self.transform(nir_img)
        else:
            rgb_img = transforms.ToTensor()(rgb_img)
            nir_img = transforms.ToTensor()(nir_img)

        img = torch.cat([rgb_img, nir_img], dim=0)
        anomaly_gt_mask = load_anomaly_masks(self.anomaly_labels_dir, filename)
        if anomaly_gt_mask is None:
            anomaly_gt_mask = torch.zeros((1, img.shape[1], img.shape[2]))
        return img, anomaly_gt_mask


def get_dataloaders(batch_size):
    data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    train_dataset = AgriVision4ChDataset(DATASET_ROOT, split='train', transform=data_transform)
    val_dataset = AgriVision4ChDataset(DATASET_ROOT, split='val', transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    return train_loader, val_loader


# --- 4. Model Definition ---
def create_swin_mae_model(mask_ratio):
    print(f"Creating Swin-Small MAE model...")
    from functools import partial
    model = swin_mae.SwinMAE(
        img_size=224, patch_size=4, in_chans=4, embed_dim=96,
        depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), window_size=7,
        mlp_ratio=4., qkv_bias=True, drop_path_rate=0.1, drop_rate=0.0,
        attn_drop_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        decoder_embed_dim=768, norm_pix_loss=False, mask_ratio=mask_ratio
    )
    return model


# --- 5. Evaluation (Standard + Manual Knee Point) ---
def evaluate_model(model, val_loader, device, epoch):
    K_ITERATIONS = 32
    model.eval()
    criterion = nn.MSELoss(reduction='mean')
    total_loss, total_iou, n_batches = 0, 0, 0

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images, masks = images.to(device), masks.to(device)
            B = images.size(0)
            recon_errors_accum = torch.zeros((B, 1, images.shape[2], images.shape[3]), device=device)
            recon_loss_accum = 0.0

            for k in range(K_ITERATIONS):
                latent, _ = model.forward_encoder(images)
                recon_patches = model.forward_decoder(latent)
                recon = model.unpatchify(recon_patches)
                recon_loss_accum += criterion(recon, images).item()
                err_map = ((images - recon) ** 2).mean(dim=1, keepdim=True)
                recon_errors_accum += err_map

            recon_errors_avg = recon_errors_accum / float(K_ITERATIONS)
            
            # --- MANUAL KNEE POINT THRESHOLDING ---
            error_map_np = recon_errors_avg.cpu().numpy()
            preds_np = np.zeros_like(error_map_np)
            
            for b in range(B): 
                err_img = error_map_np[b, 0]
                err_flat_sorted = np.sort(err_img.flatten())
                try:
                    threshold = find_knee_point_manual(err_flat_sorted)
                except Exception:
                    threshold = np.quantile(err_flat_sorted, 0.95)
                preds_np[b, 0] = (err_img >= threshold).astype(float)

            preds = torch.from_numpy(preds_np).to(device)
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = (preds + masks).sum(dim=(1, 2, 3)) - intersection
            iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()

            total_iou += iou
            total_loss += recon_loss_accum / float(K_ITERATIONS)
            n_batches += 1

    return total_loss / n_batches, total_iou / n_batches


# --- 6. Training loop (DYNAMIC MSE STAGNATION) ---
def train(model, train_loader, val_loader, device, epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) 
    best_miou = 0.0

    criterion_mse = nn.MSELoss().to(device) 
    criterion_pixelwise = nn.MSELoss(reduction='none').to(device)

    # --- DYNAMIC TRIGGER VARIABLES ---
    asl_active = False
    
    # We want to minimize validation loss (reconstruction error)
    best_warmup_val_loss = float('inf') 
    warmup_patience = 3 # Stop if error doesn't drop for 3 epochs
    warmup_patience_counter = 0

    print(f"--- Starting Training ---")
    print(f"Phase 1: Warmup with MSE. Waiting for Validation Loss to plateau (stagnate) for {warmup_patience} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")
        
        # Display Phase in Progress Bar
        phase_str = "ASL (Active)" if asl_active else "Warmup (MSE)"
        pbar.set_description(f"Epoch {epoch+1} [{phase_str}]")
        
        for images, _ in pbar:
            images = images.to(device)
            # Unsupervised: masks ignored

            latent, _ = model.forward_encoder(images)
            recon_patches = model.forward_decoder(latent)
            recon = model.unpatchify(recon_patches)
            
            loss = None
            if not asl_active:
                # Phase 1: Standard MSE
                loss = criterion_mse(recon, images)
            else:
                # Phase 2: Anomaly Suppression Loss (ASL)
                with torch.no_grad(): 
                    e_map = criterion_pixelwise(recon, images).mean(dim=1, keepdim=True)
                    max_e = torch.amax(e_map, dim=(1, 2, 3), keepdim=True)
                    w_map = max_e - e_map
                
                pixel_loss = criterion_pixelwise(recon, images).mean(dim=1, keepdim=True)
                loss = (w_map * pixel_loss).sum() / (w_map.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        # --- EVALUATION ---
        val_loss, val_miou = evaluate_model(model, val_loader, device, epoch)
        print(f"Epoch {epoch+1}/{epochs} | Phase: {phase_str} | TrainLoss={running_loss/len(train_loader):.6f} | ValLoss={val_loss:.6f} | mIoU={val_miou:.4f}")

        # --- DYNAMIC TRIGGER LOGIC (MSE STAGNATION) ---
        if not asl_active:
            # Check if Validation Loss (Reconstruction Error) improved
            if val_loss < best_warmup_val_loss:
                best_warmup_val_loss = val_loss
                warmup_patience_counter = 0 # Reset counter if we improved
                print(f"  > Val Loss improved to {val_loss:.6f}. Resetting patience.")
            else:
                warmup_patience_counter += 1
                print(f"  > Val Loss stagnated (did not improve). Patience: {warmup_patience_counter}/{warmup_patience}")
            
            # Trigger if patience runs out
            if warmup_patience_counter >= warmup_patience:
                asl_active = True
                print(f"\n---!!! DYNAMIC TRIGGER ACTIVATED !!!---")
                print(f"Validation Loss has stagnated. Model has learned the background.")
                print(f"Switching to Anomaly Suppression Loss (ASL) starting next epoch.")
                print(f"----------------------------------------\n")
        # -----------------------------------------------

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Saved best model with mIoU={best_miou:.4f}")

    print("Training complete. Best mIoU:", best_miou)


# --- 7. Main ---
def main():
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
    model = create_swin_mae_model(mask_ratio=MAE_MASK_RATIO)
    train(model, train_loader, val_loader, DEVICE, epochs=NUM_EPOCHS, lr=LEARNING_RATE)

    print("\n✅ Evaluating best saved model...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    final_loss, final_iou = evaluate_model(model, val_loader, DEVICE, epoch=0)
    print(f"\n✅ Final Evaluation | AvgLoss={final_loss:.6f} | mIoU={final_iou:.6f}")


if name == "main":
    main()
