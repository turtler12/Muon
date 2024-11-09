# Muon optimizer

This repo contains an implementation of the `Muon` optimizer described in [this thread](https://x.com/kellerjordan0/status/1842300916864844014).
Muon is the fastest known optimizer across diverse training scenarios including [CIFAR-10](https://github.com/KellerJordan/cifar10-airbench)
and [GPT-2 scale language modeling](https://github.com/KellerJordan/modded-nanogpt).

## Installation

```
pip install git+https://github.com/KellerJordan/Muon
```

## Usage

### Option 1: Internal AdamW backup

```python
from muon import Muon
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)
optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95,
                 backup_adamw_lr=3e-4, backup_adamw_betas=(0.90, 0.95),
                 backup_adamw_wd=0.01)
```

### Option 2: External AdamW backup

```python
import torch
from muon import Muon

params = set(model.parameters())
# Use Adam as fallback for <2D params and the embed/head
muon_params = set([p for p in ss if p.ndim >= 2 and p.size(0) < 10000])
other_params = params - muon_params
optimizer1 = Muon(muon_params, lr=0.02, momentum=0.95)
optimizer2 = torch.optim.AdamW(other_params, lr=3e-4, betas=(0.95, 0.95))

...

# In training loop
optimizer1.step()
optimizer2.step()
```

### Why do we need the AdamW backup?
Muon is only meant for optimizing >= 2D parameters, and it's not recommended for the embedding or classification head layers (this is similar to Shampoo and SOAP).
Therefore, you need to use a backup optimizer for those other parameters. This implementation of Muon supports an internal AdamW backup, which will automatically
be used for <2D parameters and for the embedding and classification head of a transformer (detected by assuming these have first dim >= 10000).
Alternately, you can explicitly filter the parameters and use an external backup (a separate optimizer).

## Benchmarks

For a comparison between AdamW, Shampoo, SOAP, and Muon for training a 124M-parameter transformer, see [here](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/102924_Optimizers).

## More info

See [this thread](https://x.com/kellerjordan0/status/1844782418676339059) for more info including the connection to Shampoo

## Accomplishments

* [Lowered the record for training to 94% on CIFAR-10 from 3.3 A100-seconds to 2.7 A100-seconds](https://github.com/KellerJordan/cifar10-airbench)
* [Used to train a transformer to GPT-2 (XL) performance in $175 of compute](https://x.com/kellerjordan0/status/1850995958697308307)
* [Improved the training speed record for attaining GPT-2 (small) performance by a factor of 1.35x](https://x.com/kellerjordan0/status/1842300916864844014)

