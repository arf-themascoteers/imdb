import torch
from imdb_dataset import IMDBDataset
from torch.utils.data import DataLoader


def test(device):
    batch_size = 300
    cid = IMDBDataset(is_train=False)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.MSELoss(reduction='mean')
    model = torch.load("models/machine.h5")
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    loss_cum = 0
    itr = 0
    results = []

    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        y_hat = y_hat.reshape(-1)
        loss = criterion(y_hat, y)
        itr = itr+1
        loss_cum = loss_cum + loss.item()

        for i in range(y_hat.shape[0]):
            results.append((y[i].item(), y_hat[i].item()))

    results.sort()
    gt = [i[0] for i in results]
    hat = [i[1] for i in results]
    gt = cid.unscale(gt)
    hat= cid.unscale(hat)
    for i in range(len(gt)):
        print(f"{gt[i]:.1f}\t\t\t\t{hat[i]:.1f}")

    loss_cum = loss_cum / itr
    print(f"Loss {loss_cum:.2f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(device)
