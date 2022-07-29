import torch
from age_dataset import AgeDataset
from torch.utils.data import DataLoader


def test(device):
    batch_size = 50
    cid = AgeDataset(is_train=False)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.MSELoss(reduction='mean')
    model = torch.load("models/machine.h5")
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        y_hat = y_hat.reshape(-1)
        loss = criterion(y_hat, y)
        print(f"Loss:{loss.item():.4f}")
        print("Ground Truth\t\tPredicted")
        for i in range(y_hat.shape[0]):
            gt_val = y[i]
            predicted = y_hat[i]
            print(f"{gt_val:.4f}\t\t\t\t{predicted:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(device)
