
import torch


def collate_fn(batch):
    imgs, labels  = zip(*batch)
    labels = [int(k) for k in labels]
    labels = torch.tensor(labels, dtype=torch.int64)

    #true_labels = [int(k) for k in true_labels]
    #true_labels = torch.tensor(true_labels, dtype=torch.int64)
    return torch.stack(imgs, dim=0), labels