import torch
import muon
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import contextmanager, redirect_stdout
import io
import os

################################################################ 
# Define PyTorch modules to test: MLP, CNN, GPT, TiedEmbedding #
################################################################ 

class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, 4*width)
        self.fc2 = nn.Linear(4*width, width)
        self.relu = nn.ReLU()

class CNN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(width)
        self.final_layer = nn.Linear(width, width)
        self.relu = nn.ReLU()

class Attention(nn.Module):
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)

class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = MLP(width)
        self.attention = Attention(width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        
class GPT(nn.Module):
    def __init__(self, width, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(width, width)
        self.transformer = nn.ModuleList([Block(width) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, width)

class TiedEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embedding = nn.Embedding(width, width)
        self.lm_head = nn.Linear(width, width)
        self.lm_head.weight = self.embedding.weight
        
#############################################################
# Test cases for auto-classifying Muon vs. AdamW parameters #
#############################################################

def test_mlp():
    model = MLP(width=2)
    muon_params, adamw_params = muon.get_muon_and_adamw_params(model=model)

    # check that muon_params contains the fc1 and fc2 parameters
    muon_expected = {id(model.fc1.weight), id(model.fc2.weight)}
    muon_actual = {id(p) for p in muon_params}
    if not muon_expected.issubset(muon_actual):
        print("Not all weight parameters are in muon_params")
        return False
    elif len(muon_actual - muon_expected) > 0:
        print("muon_params contains unknown extra parameters")
        return False

    # check that adamw_params contains only bias parameters
    adamw_expected = {id(model.fc1.bias), id(model.fc2.bias)}
    adamw_actual = {id(p) for p in adamw_params}
    if not adamw_expected.issubset(adamw_actual):
        print("Not all bias parameters are in adamw_params")
        return False
    elif len(adamw_actual - adamw_expected) > 0:
        print("adamw_params contains unknown extra parameters")
        return False

    return True

def test_cnn():
    model = CNN(width=2)
    muon_params, adamw_params = muon.get_muon_and_adamw_params(model=model)
    
    # check that muon_params contains the conv1, conv2, and final_layer parameters
    muon_expected = {id(model.conv1.weight), id(model.conv2.weight), id(model.final_layer.weight)}
    muon_actual = {id(p) for p in muon_params}
    if not muon_expected.issubset(muon_actual):
        print("Not all weight parameters are in muon_params")
        return False
    elif len(muon_actual - muon_expected) > 0:
        print("muon_params contains unknown extra parameters")
        return False
    
    # check that adamw_params contains the conv1, conv2, and final_layer biases, and batchnorm parameters
    adamw_expected = {id(model.conv1.bias), id(model.conv2.bias), id(model.final_layer.bias), id(model.batchnorm.weight), id(model.batchnorm.bias)}
    adamw_actual = {id(p) for p in adamw_params}
    if not adamw_expected.issubset(adamw_actual):
        print("Not all weight parameters are in adamw_params")
        return False
    elif len(adamw_actual - adamw_expected) > 0:
        print("adamw_params contains unknown extra parameters")
        return False

    return True

def test_gpt():
    model = GPT(width=2, num_layers=1)
    muon_params, adamw_params = muon.get_muon_and_adamw_params(model=model)
    
    # Check that muon_params contains weight parameters from all linear layers
    muon_expected = {
        id(model.transformer[0].attention.q_proj.weight),
        id(model.transformer[0].attention.k_proj.weight),
        id(model.transformer[0].attention.v_proj.weight), 
        id(model.transformer[0].attention.out_proj.weight),
        id(model.transformer[0].mlp.fc1.weight),
        id(model.transformer[0].mlp.fc2.weight),
        id(model.lm_head.weight),
    }
    muon_actual = {id(p) for p in muon_params}
    if not muon_expected.issubset(muon_actual):
        print("Not all weight parameters are in muon_params")
        return False
    elif len(muon_actual - muon_expected) > 0:
        print("muon_params contains unknown extra parameters")
        return False
    
    # Check that adamw_params contains the embedding weight, bias parameters, and normalization layers
    adamw_expected = {
        id(model.embedding.weight),
        id(model.lm_head.bias),
        id(model.ln.weight),
        id(model.ln.bias),
        id(model.transformer[0].mlp.fc1.bias),
        id(model.transformer[0].mlp.fc2.bias),
        id(model.transformer[0].attention.q_norm.weight),
        id(model.transformer[0].attention.q_norm.bias),
        id(model.transformer[0].attention.k_norm.weight),
        id(model.transformer[0].attention.k_norm.bias),
        id(model.transformer[0].attention.q_proj.bias),
        id(model.transformer[0].attention.k_proj.bias),
        id(model.transformer[0].attention.v_proj.bias),
        id(model.transformer[0].attention.out_proj.bias),
        id(model.transformer[0].norm1.weight),
        id(model.transformer[0].norm1.bias),
        id(model.transformer[0].norm2.weight),
        id(model.transformer[0].norm2.bias),
    }
    adamw_actual = {id(p) for p in adamw_params}
    if not adamw_expected.issubset(adamw_actual):
        print("Not all bias, embedding, or normalization parameters are in adamw_params")
        return False
    elif len(adamw_actual - adamw_expected) > 0:
        print("adamw_params contains unknown extra parameters")
        return False
    
    return True

def test_tied_embedding():
    model = TiedEmbedding(width=2)
    muon_params, adamw_params = muon.get_muon_and_adamw_params(model=model)
    
    # Check that muon_params does not contain the weight from Linear -- because it is shared with the embedding
    muon_actual = {id(p) for p in muon_params}
    if len(muon_actual) > 0:
        print("Linear weight detected in muon_params even though it is tied to the embedding")
        return False
    
    # Check that adamw_params contains the embedding weight and lm_head bias
    adamw_expected = {id(model.embedding.weight), id(model.lm_head.bias)}
    adamw_actual = {id(p) for p in adamw_params}
    if not adamw_expected.issubset(adamw_actual):
        print("Embedding weight not detected for adamw_params")
        return False
    elif len(adamw_actual - adamw_expected) > 0:
        print("adamw_params contains unknown extra parameters")
        return False
    
    return True

def test_error_messages():
    model = MLP(width=2)
    muon_params = [model.fc1.weight, model.fc2.weight]
    adamw_params = [model.fc1.weight, model.fc2.bias]
    try:
        muon.get_muon_and_adamw_params(muon_params=muon_params, adamw_params=adamw_params)
    except Exception as e:
        print(f"Threw error when passing muon_params and adamw_params correctly")
        return False
    
    try:
        muon.get_muon_and_adamw_params(model=model)
    except Exception as e:
        print("Threw error when passing model correctly")
        return False
    
    try:
        muon.get_muon_and_adamw_params(model=model, muon_params=muon_params)
    except Exception as e:
        pass
    else:
        print("Did not throw error when passing model and muon_params")
        return False
    
    try:
        muon.get_muon_and_adamw_params(model=model, params=model.parameters(), adamw_params=adamw_params)
    except Exception as e:
        pass
    else:
        print("Did not throw error when passing model, params, and adamw_params")
        return False
    
    try:
        muon.get_muon_and_adamw_params(params=model.parameters())
    except Exception as e:
        pass
    else:
        print("Did not throw error when passing params without model")
        return False
    
    # Test that passing only muon_params prints a warning, and that suppress_warning works
    @contextmanager
    def capture_stdout():
        f = io.StringIO()
        with redirect_stdout(f):
            yield f
    
    with capture_stdout() as f:
        muon.get_muon_and_adamw_params(muon_params=model.parameters())
    if "Warning" not in f.getvalue():
        print("Did not print warning when passing only muon_params")
        return False

    with capture_stdout() as f:
        muon.get_muon_and_adamw_params(muon_params=model.parameters(), suppress_warning=True)
    if f.getvalue().strip():
        print("Warning was not suppressed when suppress_warning=True")
        return False

    return True

def test_muon_step():
    model = MLP(width=2).to("cuda")
    try:
        # Initialize for a single GPU
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'  # Any free port
        dist.init_process_group(backend='nccl', rank=0, world_size=1)
        optimizer = muon.Muon(params=model.parameters(), model=model, rank=0, world_size=1)
    except Exception as e:
        print(f"Muon optimizer failed to initialize: {e}")
        return False
    
    # create data and simulate the forward pass
    random_data = torch.randn(2, 2, device="cuda")
    random_target = torch.randn(2, 2, device="cuda")
    optimizer.zero_grad()
    output = model.fc1(random_data)
    output = model.relu(output)
    output = model.fc2(output)
    loss = F.mse_loss(output, random_target)
    loss.backward()

    # check that the optimizer step throws no errors
    try:
        optimizer.step()
    except Exception as e:
        print(f"Muon optimizer failed to step: {e}")
        return False
    
    dist.destroy_process_group()
    return True

if __name__ == "__main__":
    tests = [
        test_mlp,
        test_cnn,
        test_gpt,
        test_tied_embedding,
        test_error_messages,
        test_muon_step,
    ]
    for test in tests:
        if not test():
            print(f"^^^ {test.__name__} failed")
            exit(1)
    print("All tests passed!")