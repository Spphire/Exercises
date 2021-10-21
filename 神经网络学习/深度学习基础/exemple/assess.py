from torch._C import parse_ir
from data import create_test_loader
import torch
import tqdm


def test(model, batch, device="cpu"):
    test_loader = create_test_loader(batch_size=batch)
    # model.to(device)
    model.eval()
    
    idx = torch.arange(10).to(device)

    confuse = torch.zeros([10, 10]).to(device).long()

    with torch.no_grad():
        for xs, ys in tqdm.tqdm(test_loader, desc="test"):
            inputs = xs.to(device)
            labels = ys.to(device)
            outputs = torch.argmax(model(inputs), dim=1)
            for i in range(10):
                confuse[i] += (outputs[labels == i].reshape(1, -1) == idx.reshape(-1, 1)).sum(dim=1)

    precision = confuse[idx, idx] / confuse.sum(dim=1)
    recall = confuse[idx, idx] / confuse.sum(dim=0)
    
    return {        
        "mean-precision": precision.mean().item(),
        "mean-recall": recall.mean().item(), 
        "class-precision": precision,
        "class-recall": recall,
        "total-confuse": confuse,
    }    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--batch", "-b", type=int, default=64)
    parser.add_argument("--device", "-d", default="cpu")
    opt = parser.parse_args()

    model = torch.load(opt.model).to(opt.device)
    result = test(model, opt.batch, opt.device)
    
    import matplotlib.pyplot as plt
    import seaborn
    
    precision = result["mean-precision"]
    recall = result["mean-recall"]
    confuse = result["total-confuse"].cpu().numpy()

    seaborn.heatmap(confuse, annot=True, fmt="d")
    plt.title(f"p{precision:.3f} r{recall:.3f}")
    plt.show()
