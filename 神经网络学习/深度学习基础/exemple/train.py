from model import FcModel, CnnModel, EmaModel, init_model
from data import create_test_loader, create_train_loader
from assess import test

import torch.optim as optim
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
import seaborn 

import tqdm
import os

torch.backends.cudnn.benchmark = True

def train(model, batch, epoch, lr, device="cpu"):
    os.makedirs("runs", exist_ok=True)

    train_loader = create_train_loader(batch_size=batch)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    model.to(device)
    ema_model = EmaModel(model, 0.999)

    best_criteria = 0

    bar = tqdm.trange(epoch, desc="epoch")
    for e in bar:
        model.train()
        for xs, ys in tqdm.tqdm(train_loader, desc="train"):
            # xs: [Bx1x28x28]
            # ys: [B]
            optimizer.zero_grad()
            inputs = xs.to(device)
            labels = ys.to(device)  # [B]
            labels = labels.long()
            outputs = model(inputs) # [Bx10]
            loss = loss_func(outputs, labels)
            loss.backward()     # dl/dtheta
            optimizer.step()    # update parameter
            ema_model.update(model)
        
        result = test(ema_model, batch, device)
        precision = result["mean-precision"]
        recall = result["mean-recall"]
        confuse = result["total-confuse"].cpu().numpy()
        bar.set_postfix({"precision": precision, 
                         "recall": recall})

        criteria = (precision + recall) / 2
        if criteria > best_criteria:
            best_criteria = criteria
            torch.save(ema_model, f"runs/{model.__class__.__name__}-best.pt")
        torch.save(ema_model, f"runs/{model.__class__.__name__}-last.pt")
    
    seaborn.heatmap(confuse, annot=True, fmt="d")
    plt.savefig(f"runs/{model.__class__.__name__}-confuse.jpg")


if __name__ == "__main__":
    model = FcModel()
    # model = CnnModel()
    init_model(model)
    train(model, 128, 8, 0.01, "cpu")
