# Muon: An optimizer for hidden layers in neural networks

This repo contains an implementation of the `Muon` optimizer described in [this thread](https://x.com/kellerjordan0/status/1842300916864844014) and [this writeup](https://kellerjordan.github.io/posts/muon/).

## Installation

```
pip install git+https://github.com/KellerJordan/Muon
```

## Usage

Muon is intended to optimize only the internal ≥2D parameters of a network.
Embeddings, classifier heads, and scalar or vector parameters should be optimized using AdamW.

```python
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)

from muon import Muon
# Find ≥2D parameters in the body of the network -- these will be optimized by Muon
muon_params = [p for p in model.body.parameters() if p.ndim >= 2]
# Find everything else -- these will be optimized by AdamW
adamw_params = [p for p in model.body.parameters() if p.ndim < 2] + [*model.head.parameters(), *model.embed.parameters()]
# Create the optimizer
optimizers = [Muon(muon_params, lr=0.02, momentum=0.95),
              torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)]
...

# in the training step
for opt in optimizers:
    opt.step()
```

You'll have to replace `model.body`, `model.head`, and `model.embed` with whatever subset is appropriate for your model.
E.g., for a ConvNet, `muon_params` should be all the convolutional filters, and `adamw_params` should be everything else.

## Example usage

[Usage of this Muon in the NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt/blob/d700b8724cbda3e7b1e5bcadbc0957f6ad1738fd/train_gpt.py#L519)

[Usage of a Muon variant in the CIFAR-10 speedrun](https://github.com/KellerJordan/cifar10-airbench/blob/0e6f9614572d7e8e3c259905aebc7196f91d5d79/research/clean_muon.py#L220)

## Hyperparameter tuning

Typically, the default values of momentum (0.95), nesterov (True), and ns_steps (5) work well. The only hyperparameter which must be tuned is the learning rate.

## Benchmarks

For a comparison between AdamW, Shampoo, SOAP, and Muon for training a 124M-parameter transformer, see [here](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/102924_Optimizers).

## Connection to Shampoo

See [this thread](https://x.com/kellerjordan0/status/1844782418676339059) for more info including the connection to Shampoo.

## Accomplishments

* [Lowered the record for training to 94% on CIFAR-10 from 3.3 A100-seconds to 2.7 A100-seconds](https://github.com/KellerJordan/cifar10-airbench)
* [Used to train a transformer to GPT-2 (XL) performance in $175 of compute](https://x.com/kellerjordan0/status/1850995958697308307)
* [Improved the training speed record for attaining GPT-2 (small) performance by a factor of 1.35x](https://x.com/kellerjordan0/status/1842300916864844014)

## Citation

```
@misc{jordan2024muon,
  author       = {Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and
                  Franz Cecista and Laker Newhouse and Jeremy Bernstein},
  title        = {Muon: An optimizer for hidden layers in neural networks},
  year         = {2024},
  url          = {https://kellerjordan.github.io/posts/muon/}
}
```

