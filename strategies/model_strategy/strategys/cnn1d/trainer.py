import torch


class Trainer:
    def __init__(self, model, device=None, lr=1e-3):
        self.model = model
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total = 0
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        return total_loss / max(1, total)

    def fit(self, train_loader, epochs=1, val_loader=None):
        for epoch in range(1, epochs + 1):
            loss = self.train_one_epoch(train_loader)
            print(f"Epoch {epoch}/{epochs} - train_loss: {loss:.4f}")
            if val_loader is not None:
                # simple validation pass
                self.model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        out = self.model(x)
                        preds = out.argmax(dim=1)
                        correct += (preds == y.to(self.device)).sum().item()
                        total += x.size(0)
                acc = correct / max(1, total)
                print(f"  val_acc: {acc:.4f}")
