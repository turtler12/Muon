# Muon optimizer

This repo contains an implementation of the `Muon` optimizer described in [this thread](https://x.com/kellerjordan0/status/1842300916864844014).
Muon is the fastest known optimizer across diverse training scenarios including [CIFAR-10](https://github.com/KellerJordan/cifar10-airbench)
and [GPT-2 scale language modeling](https://github.com/KellerJordan/modded-nanogpt).

## Installation

```
pip install git+https://github.com/KellerJordan/Muon
```

## Usage

Muon is intended for only the internal ≥ 2D parameters of a network. Any embedding, classifier head, or {0, 1}D parameter should be optimized using a backup optimizer instead (e.g., AdamW).
Muon provides two ways to accomplish this.

* Training a language model? Then option 1 will be fine.
* Training anything else? Use option 2 so that Muon explicitly knows about your classifier head.


### Option 1: Implicit AdamW backup

```python
from muon import Muon
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)
optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95,
                 adamw_lr=3e-4, adamw_betas=(0.90, 0.95), adamw_wd=0.01)
```

This will automatically optimize all parameters which are <2D or are detected as the embedding / lm_head using Adam.
The latter are detected by checking whether their first dim is greater than 10,000.


### Option 2: Explicit AdamW backup

```python
from muon import Muon
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)
muon_params = [p for p in model.body.parameters() if p.ndim >= 2]
adamw_params = [p for p in model.body.parameters() if p.ndim < 2]
adamw_params.extend(model.head.parameters())
adamw_params.extend(model.embed.parameters())
optimizer = Muon(muon_params, lr=0.02, momentum=0.95,
                 adamw_params=adaw_params, adamw_lr=3e-4, adamw_betas=(0.90, 0.95), adamw_wd=0.01)
```

You'll have to replace `model.body`, `model.head`, and `model.embed` with whatever subset is appropriate for your model.

## Benchmarks

For a comparison between AdamW, Shampoo, SOAP, and Muon for training a 124M-parameter transformer, see [here](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/102924_Optimizers).

## Connection to Shampoo

See [this thread](https://x.com/kellerjordan0/status/1844782418676339059) for more info including the connection to Shampoo.

## Accomplishments

* [Lowered the record for training to 94% on CIFAR-10 from 3.3 A100-seconds to 2.7 A100-seconds](https://github.com/KellerJordan/cifar10-airbench)
* [Used to train a transformer to GPT-2 (XL) performance in $175 of compute](https://x.com/kellerjordan0/status/1850995958697308307)
* [Improved the training speed record for attaining GPT-2 (small) performance by a factor of 1.35x](https://x.com/kellerjordan0/status/1842300916864844014)

