# Diagnosing Optimization During Fine-Tuning

This project compares AdamW with a hybrid Muon/AdamW optimizer while fine-tuning
`prajjwal1/bert-tiny` on the SST-2 sentiment classification dataset. It records
training and validation loss, validation accuracy, gradient norm, update norm,
and epoch time. A separate script estimates checkpoint flatness with random
parameter perturbations.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the pretrained model and SST-2 dataset from Hugging Face.

## Run

Run the optimizer comparison:

```bash
python tune.py
```

Then evaluate the flatness of the saved checkpoints:

```bash
python flatness.py
```

To evaluate the checkpoints already included in `optimization_outputs/`, run
`python flatness.py --checkpoint-dir optimization_outputs` instead.

Use `python flatness.py --help` to view optional settings such as perturbation
radii, number of random directions, and batch size.

## Outputs

- `optimization_outputs/`: training curves and the best checkpoints.
- `flatness_outputs/`: raw flatness measurements, summary statistics, and a plot.
- `report.md`: experiment setup, results, interpretation, and limitations.

The reported results are based on a single training seed and a small flatness
sample, so they should be treated as preliminary.
