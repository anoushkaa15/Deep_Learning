import torch
from tqdm import tqdm
import torch.optim as optim
from torch import nn

from model import create_swin_mae
from utils import get_loaders
from asl_loss import anomaly_suppression_loss
from evaluation import evaluate_model

DATASET_ROOT = "/path/to/Agriculture-Vision-2021"
MODEL_SAVE = "swinmae_small_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
LR = 1e-3

def train():
    train_loader, val_loader = get_loaders(DATASET_ROOT, batch=16)
    model = create_swin_mae(mask_ratio=0.75).to(DEVICE)

    mse = nn.MSELoss()
    optimz = optim.AdamW(model.parameters(), lr=LR)
    best = 0
    asl = False
    patience, counter = 3, 0
    best_val_loss = float("inf")

    for ep in range(EPOCHS):
        phase = "ASL" if asl else "MSE"
        model.train()
        epoch_loss = 0

        for imgs, _ in tqdm(train_loader, desc=f"Epoch {ep+1} [{phase}]"):
            imgs = imgs.to(DEVICE)
            latent, _ = model.forward_encoder(imgs)
            r_p = model.forward_decoder(latent)
            recon = model.unpatchify(r_p)

            loss = mse(recon, imgs) if not asl else anomaly_suppression_loss(recon, imgs)
            optimz.zero_grad()
            loss.backward()
            optimz.step()

            epoch_loss += loss.item()

        val_loss, val_iou = evaluate_model(model, val_loader, DEVICE)
        print(f"Epoch {ep+1} | {phase} | Train={epoch_loss/len(train_loader):.4f} | Val={val_loss:.4f} | mIoU={val_iou:.4f}")

        if not asl:
            if val_loss < best_val_loss:
                best_val_loss, counter = val_loss, 0
                print("  > Improved. Reset patience.")
            else:
                counter += 1
                print(f"  > Stagnated ({counter}/{patience})")
            if counter >= patience:
                asl = True
                print("\n---- Switched to ASL ----\n")

        if val_iou > best:
            best = val_iou
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"✔ Saved best model (mIoU={best:.4f})")

if __name__ == "__main__":
    train()
