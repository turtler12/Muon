import os
import torch
import torch.distributed as dist
from typing import Optional, List, Tuple
from torch import Tensor

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Muon runs an internal AdamW for any parameters that are marked adamw_params. If you don't specify
    muon_params or adamw_params, you can pass the model itself and Muon will auto-classify parameters.
    Passing the model to an optimizer isn't standard PyTorch philosophy, but Muon is an architecture-
    aware optimizer that treats linear weight parameters differently from embedding or bias parameters.
    For more on the architecture-aware approach: https://arxiv.org/abs/2410.21265

    Example usages:
    >>> # Auto-classify all the model's parameters into Muon and AdamW (default)
    >>> muon = Muon(params=model.parameters(), model=model)
    >>> # Auto-classify and optimize over specific parameters only
    >>> muon = Muon(params=params_sublist, model=model)
    >>> # Specify explicitly which should use Muon and AdamW
    >>> muon = Muon(muon_params=linear_params, adamw_params=embedding_and_head_params)
    >>> # Use two optimizers (warns you not to put embedding/head params in Muon)
    >>> muon, adamw = Muon(muon_params=muon_params), AdamW(params=adamw_params)

    Common mistake:
    >>> # This will perform poorly on the embedding/head params
    >>> muon = Muon(muon_params=model.parameters())

    Arguments:
        params: The parameters to optimize, auto-classified to use Muon or the internal AdamW.
        model: The PyTorch model, required if auto-classifying parameters into Muon and AdamW.
        muon_params: Overrides auto-classification and specifies which parameters should use Muon.
        adamw_params: Overrides auto-classification and specifies which parameters should use AdamW.
                      Any params in `muon_params` that are {0, 1}-D will be optimized by AdamW as well.
        suppress_warning: Don't warn about a common mistake (using Muon on embedding/output params)
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
        weight_decay: The weight decay for Muon parameters.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
        rank: The rank of the current GPU.    (use rank=0 for single GPU)
        world_size: The total number of GPUs. (use world_size=1 for single GPU)
    """
    def __init__(self, params=None, model=None, muon_params=None, adamw_params=None, suppress_warning=False,
                 lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.01,
                 adamw_lr=3e-4, adamw_betas=(0.95, 0.95), adamw_eps=1e-12, adamw_wd=0,
                 rank=None, world_size=None):
        if (rank is None) or (world_size is None):
            raise Exception("world_size and rank params required, if you want to use this optimizer on a single GPU, pass rank=0 and world_size=1.")
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay,
                        adamw_lr_ratio=adamw_lr/lr, adamw_betas=adamw_betas, adamw_eps=adamw_eps, adamw_wd=adamw_wd)
        muon_params, adamw_params = get_muon_and_adamw_params(
            model=model, params=params, muon_params=muon_params, adamw_params=adamw_params, suppress_warning=suppress_warning
        )
        params: list[Tensor] = [*muon_params, *adamw_params]
            
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

        # Mark parameters as Muon or AdamW
        muon_param_ids = {id(p) for p in muon_params}
        for param_group in self.param_groups:
            for p in param_group['params']:
                self.state[p]['use_muon'] = id(p) in muon_param_ids and p.ndim > 1

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################
            
            params = [p for p in group['params'] if self.state[p]['use_muon']]
            # generate weight updates in distributed fashion
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    if g.ndim == 4: # for the case of conv filters
                        g = g.view(len(g), -1)
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()
            
            ############################
            #       AdamW backup       #
            ############################
            
            params = [p for p in group['params'] if not self.state[p]['use_muon']]
            lr = group['adamw_lr_ratio'] * group['lr'] # in order for lr schedule to work
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            weight_decay = group['adamw_wd']
            
            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['moment1'] = torch.zeros_like(g)
                    state['moment2'] = torch.zeros_like(g)
                state['step'] += 1
                step = state['step']
                buf1 = state['moment1']
                buf2 = state['moment2']
                buf1.lerp_(g, 1-beta1)
                buf2.lerp_(g.square(), 1-beta2)
                
                g = buf1 / (eps + buf2.sqrt())
                
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr/scale)

def get_muon_and_adamw_params(
    model: Optional[torch.nn.Module] = None,
    params: Optional[List[torch.nn.Parameter]] = None,
    muon_params: Optional[List[torch.nn.Parameter]] = None,
    adamw_params: Optional[List[torch.nn.Parameter]] = None,
    suppress_warning: bool = False,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Auto-classifies parameters into two groups:
    1. Linear/conv weight parameters (for Muon)
    2. Bias, embedding, and other (for AdamW)

    To directly specify, pass muon_params and adamw_params, and do not set model or params.

    Arguments:
        model: full model required if auto-classifying parameters (default)
        params: specific parameters to auto-classify into either Muon or AdamW
        muon_params: overrides auto-classification and specifies which parameters should use Muon
        adamw_params: overrides auto-classification and specifies which parameters should use AdamW
        suppress_warning: don't warn about a common mistake (using Muon on embedding/output params)
    """
    assert (model is None and params is None) or (muon_params is None and adamw_params is None), (
        "Cannot mix auto-classifying (params/model) with explicit parameter grouping (muon_params/adamw_params). For manual control, do not pass model or params."
    )
    if muon_params is not None and adamw_params is not None:
        return muon_params, adamw_params
    elif muon_params is not None:
        if not suppress_warning:
            print(
                "Warning: you are optimizing all parameters with Muon, but did you mean to use the default optimizer "
                "AdamW for bias or embedding parameters? To ignore this message, you can set suppress_warning=True."
            )
        return muon_params, []
    elif adamw_params is not None:
        return [], adamw_params
    elif model is not None:
        # Auto-classify into Muon or AdamW based on whether a param is a hidden layer parameter
        params = params if params is not None else list(model.parameters())
        embedding_ids = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
        def muon_criterion(module, param):
            return (
                param.ndim > 1 and
                id(param) not in embedding_ids and  # prevents misclassifying tied embedding weights
                isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d))
            )
        muon_param_ids = set()
        for module in model.modules():
            for param in module.parameters(recurse=False):
                if muon_criterion(module, param):
                    muon_param_ids.add(id(param))
        muon_params = [p for p in params if id(p) in muon_param_ids]
        adamw_params = [p for p in params if id(p) not in muon_param_ids]
        return muon_params, adamw_params
    else:
        raise Exception(
            "To auto-classify params to use Muon or its built-in AdamW, you need to pass the model like Muon(..., model=model). "
            "If you want direct control over which params use Muon or AdamW, pass muon_params and adamw_params instead."
        )