from torch.utils.data import DataLoader
from .dataset import AgriVision4ChDataset

def get_loaders(root, batch=16):
    train = AgriVision4ChDataset(root, split='train')
    val = AgriVision4ChDataset(root, split='val')
    return (
        DataLoader(train, batch_size=batch, shuffle=True, num_workers=2, drop_last=True),
        DataLoader(val, batch_size=batch, shuffle=False, num_workers=2, drop_last=True)
    )
