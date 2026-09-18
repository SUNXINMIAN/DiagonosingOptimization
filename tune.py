import torch
import random
import numpy as np
from torch.optim import AdamW
from muon import SingleDeviceMuon
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding
import copy
import time
import matplotlib.pyplot as plt
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class HybridMuonAdamW:
    """Use Muon for encoder matrices and AdamW for all other parameters."""

    def __init__(self, model, lr):
        muon_params = []
        adamw_params = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim >= 2 and name.startswith("bert.encoder."):
                muon_params.append(parameter)
            else:
                adamw_params.append(parameter)

        self.optimizers = [
            SingleDeviceMuon(muon_params, lr=lr),
            AdamW(adamw_params, lr=lr),
        ]

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

def train(model, optimizer, train_loader, val_loader, optimizer_name):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "grad_norm": [],
        "update_norm": [],
        "epoch_time": [],
    }
    best_acc = 0.0
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        epoch_grad_norm = 0.0
        epoch_update_norm = 0.0
        start_time = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            epoch_grad_norm += grad_norm

            old_params = [p.detach().clone() for p in model.parameters()]
            optimizer.step()
            update_norm = 0.0
            for old_p, new_p in zip(old_params, model.parameters()):
                update_norm += ((new_p.detach() - old_p).norm(2).item()**2)
            update_norm = update_norm ** 0.5
            epoch_update_norm += update_norm

            train_loss += loss.item()
        epoch_time = time.time() - start_time
        train_loss /= len(train_loader)
        epoch_grad_norm /= len(train_loader)
        epoch_update_norm /= len(train_loader)
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)
                loss = outputs.loss
                val_loss += loss.item()
                predictions = outputs.logits.argmax(dim=-1)
                correct += (predictions == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        accuracy = correct / total
        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(accuracy)
        history["grad_norm"].append(epoch_grad_norm)
        history["update_norm"].append(epoch_update_norm)
        history["epoch_time"].append(epoch_time)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), f"best_{optimizer_name}.pth")

        print(f"[{optimizer_name}] \
            Epoch {epoch+1}, \
            Train Loss: {train_loss:.4f}, \
            Val Loss: {val_loss:.4f}, \
            Val Acc: {accuracy:.4f}, \
            Grad Norm: {epoch_grad_norm:.4f}, \
            Update Norm: {epoch_update_norm:.6f}, \
            Time: {epoch_time:.2f} \
        ")
    print(f"[{optimizer_name}] Best Val Acc: {best_acc:.4f}")
    return history

def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=128)

def plot(history_muon, history_adamw):
    epochs = range(1, len(history_muon["train_loss"]) + 1)
    metrics = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Validation Loss"),
        ("val_acc", "Validation Accuracy"),
        ("grad_norm", "Gradient Norm"),
        ("update_norm", "Update Norm"),
        ("epoch_time", "Epoch Time (s)"),
    ]

    for key, name in metrics:
        plt.figure(figsize=(6,4))
        plt.plot(epochs, history_muon[key], marker="o", label="Muon",)
        plt.plot(epochs, history_adamw[key], marker="s", label="AdamW",)

        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.title(f"{name}: Muon vs AdamW")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"optimization_outputs/{key}.png", dpi=300)
        #plt.show()

if __name__ == "__main__":
    set_seed(42)
    model_name = "prajjwal1/bert-tiny"
    dataset = load_dataset("stanfordnlp/sst2")

    vocab_path = hf_hub_download(repo_id=model_name, filename="vocab.txt")
    try:
        tokenizer = BertTokenizer(vocab=vocab_path)
    except TypeError:
        tokenizer = BertTokenizer(vocab_file=vocab_path)
        
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.remove_columns(["sentence"])
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(dataset["validation"], batch_size=16, shuffle=False, collate_fn=collator)
    #test_loader = DataLoader(dataset["test"], batch_size=16, shuffle=False, collate_fn=collator)

    base_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    model_muon = copy.deepcopy(base_model)
    optimizer_muon = HybridMuonAdamW(model_muon, lr=2e-5)
    history_muon = train(model_muon, optimizer_muon, train_loader, val_loader, "Muon")
    
    model_adamw = copy.deepcopy(base_model)
    optimizer_adamw = AdamW(model_adamw.parameters(), lr=2e-5)
    history_adamw = train(model_adamw, optimizer_adamw, train_loader, val_loader, "AdamW")

    plot(history_muon, history_adamw)
