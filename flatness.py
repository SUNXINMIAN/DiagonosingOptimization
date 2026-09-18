"""Estimate checkpoint flatness with small random parameter perturbations."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
)


MODEL_NAME = "prajjwal1/bert-tiny"
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare checkpoint flatness using random parameter perturbations."
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[1e-3, 5e-3, 1e-2],
        help="Perturbation norms as fractions of the model parameter norm.",
    )
    parser.add_argument(
        "--directions",
        type=int,
        default=5,
        help="Number of random directions tested at each radius.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Maximum validation batches per evaluation; 0 uses all batches.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing best_Muon.pth and best_AdamW.pth. "
        "By default, the script checks its own directory and outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "flatness_outputs",
    )
    return parser.parse_args()


def load_tokenizer():
    """Load the old-style bert-tiny vocabulary across Transformers versions."""
    vocab_path = hf_hub_download(repo_id=MODEL_NAME, filename="vocab.txt")
    try:
        return BertTokenizer(vocab=vocab_path)
    except TypeError:
        return BertTokenizer(vocab_file=vocab_path)


def make_validation_loader(tokenizer, batch_size):
    validation = load_dataset("stanfordnlp/sst2", split="validation")

    def tokenize(batch):
        encoded = tokenizer(batch["sentence"], truncation=True, max_length=128)
        encoded["labels"] = batch["label"]
        return encoded

    validation = validation.map(
        tokenize,
        batched=True,
        remove_columns=validation.column_names,
    )
    validation.set_format("torch")
    return DataLoader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )


def load_checkpoint(path, device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    config = BertConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = BertForSequenceClassification(config)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device)


def find_checkpoint(filename, checkpoint_dir=None):
    candidates = []
    if checkpoint_dir is not None:
        candidates.append(checkpoint_dir / filename)
    candidates.extend([SCRIPT_DIR / filename, SCRIPT_DIR / "outputs" / filename])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Checked:\n  {checked}")


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    loss_sum = 0.0
    correct = 0
    example_count = 0

    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = {name: value.to(device) for name, value in batch.items()}
        outputs = model(**batch)
        batch_size = batch["labels"].shape[0]
        loss_sum += outputs.loss.item() * batch_size
        correct += (outputs.logits.argmax(dim=-1) == batch["labels"]).sum().item()
        example_count += batch_size

    if example_count == 0:
        raise ValueError("No validation examples were evaluated.")
    return loss_sum / example_count, correct / example_count


def perturbation_for(model, relative_radius, seed):
    """Create delta with ||delta||_2 = relative_radius * ||parameters||_2."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_norm = math.sqrt(
        sum(parameter.detach().float().pow(2).sum().item() for parameter in parameters)
    )

    generator = torch.Generator(device=parameters[0].device)
    generator.manual_seed(seed)
    directions = [
        torch.randn(
            parameter.shape,
            dtype=torch.float32,
            device=parameter.device,
            generator=generator,
        )
        for parameter in parameters
    ]
    direction_norm = math.sqrt(
        sum(direction.pow(2).sum().item() for direction in directions)
    )
    scale = relative_radius * parameter_norm / direction_norm
    deltas = [
        direction.mul(scale).to(dtype=parameter.dtype)
        for direction, parameter in zip(directions, parameters)
    ]
    return parameters, deltas


@torch.no_grad()
def apply_perturbation(parameters, deltas, sign=1.0):
    for parameter, delta in zip(parameters, deltas):
        parameter.add_(delta, alpha=sign)


def measure_checkpoint(
    name,
    checkpoint,
    loader,
    device,
    radii,
    directions,
    seed,
    max_batches,
):
    model = load_checkpoint(checkpoint, device)
    baseline_loss, baseline_accuracy = evaluate(model, loader, device, max_batches)
    print(
        f"{name}: baseline loss={baseline_loss:.6f}, "
        f"accuracy={baseline_accuracy:.4f}"
    )

    rows = []
    for radius in radii:
        for direction_index in range(directions):
            # Reusing these seeds gives both checkpoints the same random directions.
            direction_seed = seed + direction_index
            parameters, deltas = perturbation_for(model, radius, direction_seed)
            apply_perturbation(parameters, deltas)
            try:
                positive_loss, positive_accuracy = evaluate(
                    model, loader, device, max_batches
                )
            finally:
                apply_perturbation(parameters, deltas, sign=-1.0)

            apply_perturbation(parameters, deltas, sign=-1.0)
            try:
                negative_loss, negative_accuracy = evaluate(
                    model, loader, device, max_batches
                )
            finally:
                apply_perturbation(parameters, deltas)

            if positive_loss >= negative_loss:
                worst_loss, worst_accuracy = positive_loss, positive_accuracy
            else:
                worst_loss, worst_accuracy = negative_loss, negative_accuracy

            rows.append(
                {
                    "optimizer": name,
                    "radius": radius,
                    "direction": direction_index,
                    "baseline_loss": baseline_loss,
                    "positive_loss": positive_loss,
                    "negative_loss": negative_loss,
                    "worst_loss": worst_loss,
                    "loss_increase": worst_loss - baseline_loss,
                    "baseline_accuracy": baseline_accuracy,
                    "worst_accuracy": worst_accuracy,
                }
            )
    return rows


def summarize(rows):
    summaries = []
    keys = sorted({(row["optimizer"], row["radius"]) for row in rows})
    for optimizer, radius in keys:
        values = [
            row["loss_increase"]
            for row in rows
            if row["optimizer"] == optimizer and row["radius"] == radius
        ]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summaries.append(
            {
                "optimizer": optimizer,
                "radius": radius,
                "mean_loss_increase": mean,
                "std_loss_increase": math.sqrt(variance),
                "max_loss_increase": max(values),
            }
        )
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path, summaries):
    plt.figure(figsize=(6, 4))
    for optimizer in ("Muon", "AdamW"):
        selected = [row for row in summaries if row["optimizer"] == optimizer]
        selected.sort(key=lambda row: row["radius"])
        plt.errorbar(
            [row["radius"] for row in selected],
            [row["mean_loss_increase"] for row in selected],
            yerr=[row["std_loss_increase"] for row in selected],
            marker="o",
            capsize=3,
            label=optimizer,
        )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Relative perturbation radius")
    plt.ylabel("Validation loss increase")
    plt.title("Random-perturbation flatness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    if args.directions < 1:
        raise ValueError("--directions must be at least 1")
    if any(radius <= 0 for radius in args.radii):
        raise ValueError("All perturbation radii must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = load_tokenizer()
    loader = make_validation_loader(tokenizer, args.batch_size)

    checkpoints = {
        "Muon": find_checkpoint("best_Muon.pth", args.checkpoint_dir),
        "AdamW": find_checkpoint("best_AdamW.pth", args.checkpoint_dir),
    }
    rows = []
    for name, checkpoint in checkpoints.items():
        rows.extend(
            measure_checkpoint(
                name,
                checkpoint,
                loader,
                device,
                args.radii,
                args.directions,
                args.seed,
                args.max_batches,
            )
        )

    summaries = summarize(rows)
    print("\nMean loss increase (smaller is flatter):")
    for row in summaries:
        print(
            f"{row['optimizer']:5s}  radius={row['radius']:.4g}  "
            f"mean={row['mean_loss_increase']:+.6f}  "
            f"std={row['std_loss_increase']:.6f}  "
            f"max={row['max_loss_increase']:+.6f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "flatness_raw.csv", rows)
    write_csv(args.output_dir / "flatness_summary.csv", summaries)
    plot_summary(args.output_dir / "flatness.png", summaries)
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
