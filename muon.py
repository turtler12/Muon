
#%%
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch
from matplotlib.collections import LineCollection
import numpy as np


#%%
def zeropower_via_newtonschulz5(G, steps: int):
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

    


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim < 2:
        return update  # Skip orthogonalization for 1D params
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


#%%


def muon_on_linear_regression_scenario():

    # Simple linear model
    model = torch.nn.Linear(10, 5)
    print(model)
    criterion = torch.nn.MSELoss()
    print(criterion)

    # Create optimizer
    optimizer = SingleDeviceMuon(list(model.parameters()), lr=0.01, momentum=0.9)

    # Dummy input and target
    x = torch.randn(4, 10)
    target = torch.randn(4, 5)

    # Training step
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print("Loss:", loss.item())
    return 

def muon_on_simple_dome():
    from muon import SingleDeviceMuon

    # global minima at (2, -3)
    def dome_function(params):
        x, y = params[0], params[1]
        return (x-2)**2 + (y+3)**2
    params = torch.nn.Parameter(torch.randn(2), requires_grad = True)
    optimizer = SingleDeviceMuon([params], lr=0.1, momentum=0.95)
    
    losses = []
    positions = []

    for step in range(100):
        optimizer.zero_grad()
        loss = dome_function(params)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        positions.append(params.detach().cpu().numpy().copy())

    print("Overall loss: ", losses[-1])

    positions = torch.tensor(positions)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss over steps")

    # Gradient coloring for trajectory
    pts = positions.numpy()
    points = pts.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(segments))
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(np.arange(len(segments)))
    lc.set_linewidth(2)

    plt.subplot(1,2,2)
    # Plot dome gradient field
    xg, yg = np.meshgrid(
        np.linspace(pts[:,0].min()-1, pts[:,0].max()+1, 20),
        np.linspace(pts[:,1].min()-1, pts[:,1].max()+1, 20)
    )
    grad_x = 2 * (xg - 2)
    grad_y = 2 * (yg + 3)
    plt.quiver(xg, yg, -grad_x, -grad_y, color='lightgray', alpha=0.6, width=0.003, scale=30, label='Dome Gradient')

    plt.gca().add_collection(lc)
    plt.scatter([2], [-3], color='red', label='Global Min', zorder=3)
    plt.scatter(pts[0,0], pts[0,1], color='orange', label='Start', zorder=3)
    plt.scatter(pts[-1,0], pts[-1,1], color='blue', label='End', zorder=3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Parameter trajectory")
    plt.legend()
    plt.xlim(pts[:,0].min()-1, pts[:,0].max()+1)
    plt.ylim(pts[:,1].min()-1, pts[:,1].max()+1)
    plt.tight_layout()
    plt.show()
    return 

def muon_logistic_regression():
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    # generating synthetic binary classification data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # basic features features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Define logistic regression model
    # just a simple logistic regression model with linear layer followed by sigmoid activation
    model = torch.nn.Sequential(
        torch.nn.Linear(20, 1),
        torch.nn.Sigmoid()
    )

    # Define loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = SingleDeviceMuon(list(model.parameters()), lr=0.01, momentum=0.9)

    # Training loop
    num_epochs = 100
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_preds = (val_outputs >= 0.5).float()
            val_accuracy = accuracy_score(y_val_tensor.numpy(), val_preds.numpy())
            val_losses.append(val_loss.item())
            val_accuracies.append(val_accuracy)

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}')

    # Plot loss and val accuracy
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(val_accuracies, label='Val Accuracy', color='green')
    ax2.set_ylabel('Val Accuracy')
    ax2.legend(loc='upper right')

    plt.title('Muon Logistic Regression: Loss & Validation Accuracy')
    plt.tight_layout()
    plt.show()

    return

def compare_muon_with_adam_logistic_regression():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    # Generate synthetic binary classification data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Define two identical models
    def get_model():
        return torch.nn.Sequential(
            torch.nn.Linear(20, 1),
            torch.nn.Sigmoid()
        )

    model_muon = get_model()
    model_adam = get_model()

    criterion = torch.nn.BCELoss()
    optimizer_muon = SingleDeviceMuon(list(model_muon.parameters()), lr=0.01, momentum=0.99)
    optimizer_adam = torch.optim.Adam(list(model_adam.parameters()), lr=0.01)

    num_epochs = 100
    train_losses_muon, val_losses_muon, val_accuracies_muon = [], [], []
    train_losses_adam, val_losses_adam, val_accuracies_adam = [], [], []

    for epoch in range(num_epochs):
        # Muon
        model_muon.train()
        optimizer_muon.zero_grad()
        outputs_muon = model_muon(X_train_tensor)
        loss_muon = criterion(outputs_muon, y_train_tensor)
        loss_muon.backward()
        optimizer_muon.step()
        train_losses_muon.append(loss_muon.item())

        model_muon.eval()
        with torch.no_grad():
            val_outputs_muon = model_muon(X_val_tensor)
            val_loss_muon = criterion(val_outputs_muon, y_val_tensor)
            val_preds_muon = (val_outputs_muon >= 0.5).float()
            val_accuracy_muon = accuracy_score(y_val_tensor.numpy(), val_preds_muon.numpy())
            val_losses_muon.append(val_loss_muon.item())
            val_accuracies_muon.append(val_accuracy_muon)

        # Adam
        model_adam.train()
        optimizer_adam.zero_grad()
        outputs_adam = model_adam(X_train_tensor)
        loss_adam = criterion(outputs_adam, y_train_tensor)
        loss_adam.backward()
        optimizer_adam.step()
        train_losses_adam.append(loss_adam.item())

        model_adam.eval()
        with torch.no_grad():
            val_outputs_adam = model_adam(X_val_tensor)
            val_loss_adam = criterion(val_outputs_adam, y_val_tensor)
            val_preds_adam = (val_outputs_adam >= 0.5).float()
            val_accuracy_adam = accuracy_score(y_val_tensor.numpy(), val_preds_adam.numpy())
            val_losses_adam.append(val_loss_adam.item())
            val_accuracies_adam.append(val_accuracy_adam)

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] | Muon Loss: {loss_muon.item():.4f}, Val Acc: {val_accuracy_muon:.4f} | Adam Loss: {loss_adam.item():.4f}, Val Acc: {val_accuracy_adam:.4f}")')

    # Plot comparison
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(train_losses_muon, label='Muon Train Loss', color='blue')
    ax1.plot(val_losses_muon, label='Muon Val Loss', color='cyan')
    ax1.plot(train_losses_adam, label='Adam Train Loss', color='orange')
    ax1.plot(val_losses_adam, label='Adam Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(val_accuracies_muon, label='Muon Val Accuracy', color='green')
    ax2.plot(val_accuracies_adam, label='Adam Val Accuracy', color='purple')
    ax2.set_ylabel('Val Accuracy')
    ax2.legend(loc='upper right')

    plt.title('Muon vs Adam: Loss & Validation Accuracy')
    plt.tight_layout()
    plt.show()

    return

#%%
if __name__ == "__main__":
    #muon_on_linear_regression_scenario()
    #muon_on_simple_dome()
    #muon_logistic_regression()
    compare_muon_with_adam_logistic_regression()
# %%