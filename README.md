# Muon optimizer

This repo contains an implementation of the `Muon` optimizer described in [this thread](https://x.com/kellerjordan0/status/1842300916864844014).
Muon is the fastest known optimizer across diverse scenarios including [CIFAR-10](https://github.com/KellerJordan/cifar10-airbench)
and [GPT-2](https://github.com/KellerJordan/modded-nanogpt) training.


## Usage

```python
import torch
from muon import Muon

params = set(model.parameters())
# Use Adam as fallback for <2D params and the embed/head
muon_params = set([p for p in ss if p.ndim >= 2 and p.size(0) < 10000])
other_params = params - matrix_params
optimizer1 = Muon(muon_params, lr=0.02,  momentum=0.95)
optimizer2 = torch.optim.Adam(other_params, lr=3e-4, betas=(0.95, 0.95))

...

# In training loop
optimizer1.step()
optimizer2.step()
```

## Accomplishments

* [Lowered the record for training to 94% on CIFAR-10 from 3.3 A100-seconds to 2.7 A100-seconds](https://github.com/KellerJordan/cifar10-airbench)
* [Used to train a transformer to GPT-2 (XL) performance in $175 of compute](https://x.com/kellerjordan0/status/1850995958697308307)
* [Improved the training speed record for attaining GPT-2 (small) performance by a factor of 1.35x](https://x.com/kellerjordan0/status/1842300916864844014)

