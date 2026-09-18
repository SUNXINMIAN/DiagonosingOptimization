# AdamW vs. Muon on SST-2

## Setup and measurements

I fine-tuned `prajjwal1/bert-tiny` on SST-2 for 10 epochs with batch size 16, maximum length 128, and learning rate `2e-5`. Both runs started from the same pretrained weights. AdamW updated every parameter. Muon updated the 2-D encoder weights and AdamW handled the remaining parameters because Muon is not intended for vectors such as biases.

I used training loss, validation loss, validation accuracy, gradient norm, update norm, and epoch time to compare performance, stability, step size, and cost.

## Optimization results

| Measurement | Muon | AdamW |
|---|---:|---:|
| Best validation accuracy | 82.80% (epoch 4) | 82.68% (epoch 4) |
| Lowest validation loss | 0.4323 (epoch 2) | 0.4281 (epoch 2) |
| Mean gradient norm | 6.2309 | 6.3590 |
| Mean update norm | 0.004099 | 0.004863 |
| Mean epoch time | 328.1 s | 231.4 s |

Muon's best validation accuracy was 82.80%, while AdamW's best validation accuracy was 82.68%. The difference is very small. AdamW had a slightly lower validation loss and was faster. Muon needed about 42% more time per epoch. The mean gradient norms were close, so there isn't a clear difference in gradient stability. Muon's mean update norm was slightly smaller, which means that it made slightly more conservative parameter updates. Both models started to overfit after epoch 4: training loss went down, but validation loss went up.

## Flatness

I wrote the training code myself and used GPT to help write a separate flatness test. The script adds random noise to each best checkpoint and measures the change in validation loss. I used three noise levels, equal to 0.1%, 0.5%, and 1% of the size of all model parameters taken together, with five random directions per level. For each direction, the script tries adding and subtracting the noise and keeps the higher loss. A smaller loss increase means a flatter result.

| Noise level | Muon mean / largest loss increase | AdamW mean / largest loss increase |
|---:|---:|---:|
| 0.001 | 0.000070 / 0.000204 | 0.000156 / 0.000609 |
| 0.005 | 0.000366 / 0.001103 | 0.000819 / 0.003190 |
| 0.010 | 0.000768 / 0.002417 | 0.001732 / 0.006748 |

Muon was less sensitive at every noise level. Its mean loss increase was about 55% lower, and its largest increase was 64-67% lower. Based on this test, the Muon checkpoint appears flatter.

## Reliability and a larger study

These results are only preliminary. I used one training seed and five random directions, so another run could give different results. I also used the same learning rate for both optimizers without tuning it.

In a larger study, I would use more random seeds and tune more hyperparameters, while giving both optimizers the same search budget. I would also repeat the comparison across different model sizes, tasks, and dataset sizes and types.

Run the experiments with `python tune.py` and `python flatness.py`. Optimization results are saved in `optimization_outputs/`. Flatness results are saved in `flatness_outputs/`.
