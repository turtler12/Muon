## To use:


```python
import torch
from muon import Muon

params = set(model.parameters())
# Use Adam as fallback for <2D params and the embed/head
muon_params = set([p for p in ss if p.ndim == 2 and p.size(0) < 10000])
other_params = params - matrix_params
optimizer1 = Muon(muon_params, lr=0.02,  momentum=0.95)
optimizer2 = torch.optim.Adam(other_params, lr=3e-4, betas=(0.95, 0.95))

...

# In training loop
optimizer1.step()
optimizer2.step()
```
