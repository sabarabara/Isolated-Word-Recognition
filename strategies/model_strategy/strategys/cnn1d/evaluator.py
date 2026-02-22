import torch


class Evaluator:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)

        return {"accuracy": correct / max(1, total)}
