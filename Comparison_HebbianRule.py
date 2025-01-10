import typing
import warnings
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

###############################################################################
# 1) PCLayer
###############################################################################
class PCLayer(nn.Module):
    """
    Predictive Coding layer that holds a latent variable x and computes an energy 
    based on (mu - x)^2 or another error function. If in training mode, it creates 
    x on forward when set_is_sample_x(True).
    """
    def __init__(
        self,
        energy_fn=lambda inputs: 0.5 * (inputs['mu'] - inputs['x'])**2,
        energy_fn_kwargs={},
        sample_x_fn=lambda inputs: inputs['mu'].detach().clone(),
        S=None,
        M=None,
        is_holding_error=False,
        is_keep_energy_per_datapoint=False,
    ):
        super().__init__()
        self._energy_fn = energy_fn
        self._energy_fn_kwargs = energy_fn_kwargs
        self._sample_x_fn = sample_x_fn
        self.is_holding_error = is_holding_error
        self.is_keep_energy_per_datapoint = is_keep_energy_per_datapoint
        self._S = S
        self._M = M
        self._is_sample_x = False
        self._x = None

        self.clear_energy()
        if self.is_keep_energy_per_datapoint:
            self.clear_energy_per_datapoint()
        self.eval()

    def set_is_sample_x(self, is_sample_x):
        self._is_sample_x = is_sample_x

    def get_x(self):
        return self._x

    def energy(self):
        return self._energy

    def clear_energy(self):
        self._energy = None

    def energy_per_datapoint(self):
        assert self.is_keep_energy_per_datapoint
        return self._energy_per_datapoint

    def clear_energy_per_datapoint(self):
        if self.is_keep_energy_per_datapoint:
            self._energy_per_datapoint = None

    def forward(self, mu, energy_fn_additional_inputs={}):
        # If eval mode, skip creating x
        if not self.training:
            return mu

        # Decide if we need to create x
        if not self._is_sample_x:
            self._is_sample_x = (
                self._x is None 
                or mu.device != self._x.device 
                or mu.size() != self._x.size()
            )

        if self._is_sample_x:
            # Create the latent variable parameter
            self._x = nn.Parameter(
                self._sample_x_fn({'mu': mu, 'x': self._x}).to(mu.device),
                requires_grad=True
            )
            self._is_sample_x = False

        x = self._x
        energy_fn_inputs = {**{'mu': mu, 'x': x}, **energy_fn_additional_inputs}

        # If we have S or M for structured energies, handle shape expansions
        if self._S is not None:
            assert mu.dim() == 2 and x.dim() == 2
            assert self._S.size() == (mu.size(1), x.size(1))
            mu = mu.unsqueeze(2).expand(-1, -1, x.size(1))
            x = x.unsqueeze(1).expand(-1, mu.size(1), -1)

        # Compute the energy
        energy = self._energy_fn(energy_fn_inputs, **self._energy_fn_kwargs)
        if self._S is not None:
            energy = energy * self._S.unsqueeze(0)
        elif self._M is not None:
            energy = energy * self._M.unsqueeze(0)

        if self.is_keep_energy_per_datapoint:
            self._energy_per_datapoint = energy.sum(dim=list(range(1, energy.dim()))).unsqueeze(1)

        self._energy = energy.sum()
        if self.is_holding_error:
            self.error = (self._x.data - mu).detach().clone()

        return self._x


###############################################################################
# 2) HebbianLayer
###############################################################################
class HebbianLayer(nn.Module):
    """
    A custom layer that performs a linear transformation and can do a local 
    Hebbian update on its weights after forward passes.
    """
    
    def __init__(self, in_features, out_features, bias=True, lr_hebb=1e-4):
        super().__init__()
        self.lr_hebb = lr_hebb  # Hebbian learning rate
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        self.pre_syn = x.detach()  # store for Hebbian
        out = x @ self.weight.t()
        if self.bias is not None:
            out += self.bias
        self.post_syn = out.detach()
        return out

    def hebbian_update(self):
        """
        Simple correlation-based Hebbian rule:
            dW_ij = lr_hebb * mean(post_j * pre_i)
        """
        if hasattr(self, "pre_syn") and hasattr(self, "post_syn"):
            batch_size = self.pre_syn.shape[0]
            deltaW = torch.einsum('bo,bi->oi', self.post_syn, self.pre_syn) / batch_size
            self.weight.data += self.lr_hebb * deltaW
            
            # Regularization: Clip weights to avoid excessive growth
            self.weight.data = torch.clamp(self.weight.data, -1.0, 1.0)
            
            # Optional: Apply weight decay
            self.weight.data *= (1 - self.lr_hebb * 0.01)  # 0.01 is the decay rate

###############################################################################
# 3) PCTrainer - for networks containing PCLayer
###############################################################################
class PCTrainer:
    """
    A trainer for networks that have PCLayers. We do T steps of latent update 
    and also can do an optional Hebbian update if the network contains HebbianLayers.
    """
    def __init__(
        self,
        model: nn.Module,
        T=20,
        optimizer_x_fn=optim.SGD,
        optimizer_x_kwargs={"lr": 0.01},
        optimizer_p_fn=optim.Adam,
        optimizer_p_kwargs={"lr": 0.001},
        update_x_at="all",
        update_p_at="last",
        energy_coefficient=1.0,
        do_hebbian_update=False
    ):
        self.model = model
        self.T = T
        self.optimizer_x_fn = optimizer_x_fn
        self.optimizer_x_kwargs = optimizer_x_kwargs
        self.optimizer_p_fn = optimizer_p_fn
        self.optimizer_p_kwargs = optimizer_p_kwargs

        self.update_x_at = self._parse_steps(update_x_at, T)
        self.update_p_at = self._parse_steps(update_p_at, T)
        self.energy_coefficient = energy_coefficient
        self.do_hebbian_update = do_hebbian_update

        self.optimizer_x = None
        self.optimizer_p = None

        self._create_optimizer_p()

    def _create_optimizer_p(self):
        """Create the parameter optimizer (excluding x)."""
        p_list = list(self.get_model_parameters())
        self.optimizer_p = self.optimizer_p_fn(p_list, **self.optimizer_p_kwargs)

    def _create_optimizer_x(self):
        """Create the x optimizer (for the latent states)."""
        x_list = list(self.get_model_xs())
        if len(x_list) == 0:
            # No x found, skip
            self.optimizer_x = None
            return
        self.optimizer_x = self.optimizer_x_fn(x_list, **self.optimizer_x_kwargs)

    def get_model_pc_layers(self):
        for m in self.model.modules():
            if isinstance(m, PCLayer):
                yield m

    def has_pc_layer(self):
        return any(True for _ in self.get_model_pc_layers())

    def get_model_xs(self):
        """All x from all PCLayers."""
        for m in self.get_model_pc_layers():
            x = m.get_x()
            if x is not None:
                yield x

    def get_model_parameters(self):
        """All trainable parameters except the PCLayer 'x' parameters."""
        xs = set(self.get_model_xs())
        for p in self.model.parameters():
            if p not in xs:
                yield p

    def get_energies(self):
        """Sum energies from all PCLayers."""
        total = 0.0
        found = False
        for m in self.get_model_pc_layers():
            E = m.energy()
            if E is not None:
                total += E
                found = True

        return total

    def train_on_batch(self, data, loss_fn, loss_fn_kwargs=None):
        if loss_fn_kwargs is None:
            loss_fn_kwargs = {}


        # (1) On the first iteration, tell each PC layer to sample x
        #     Then do a forward pass to ensure x is created
        for pc in self.get_model_pc_layers():
            pc.set_is_sample_x(True)

        out = self.model(data)
        # Now we create x-optimizer
        self._create_optimizer_x()

        # do T steps
        energy_val = 0.0
        sup_loss_val = 0.0

        for t in range(self.T):
            outputs = self.model(data)  # forward
            energy = self.get_energies()  
            sup_loss = loss_fn(outputs, **loss_fn_kwargs)  # supervised loss, e.g. MSE
            total_loss = energy * self.energy_coefficient + sup_loss

            # zero grads if we update x this step
            if self.optimizer_x and t in self.update_x_at:
                self.optimizer_x.zero_grad()

            # zero grads if we update p this step
            if t in self.update_p_at:
                self.optimizer_p.zero_grad()

            total_loss.backward()

            # step x
            if self.optimizer_x and t in self.update_x_at:
                self.optimizer_x.step()

            # step p
            if t in self.update_p_at:
                self.optimizer_p.step()

            energy_val = energy.item()
            sup_loss_val = sup_loss.item()

        # after T steps, optionally do a Hebbian update
        if self.do_hebbian_update:
            for module in self.model.modules():
                if hasattr(module, "hebbian_update"):
                    module.hebbian_update()

        return energy_val, sup_loss_val

    def _parse_steps(self, step_option, T):
        """String -> list-of-int converter."""
        if isinstance(step_option, str):
            dct = {
                "all": list(range(T)),
                "last": [T - 1],
                "never": [],
                "last_half": list(range(T//2, T))
            }
            return dct.get(step_option, [])
        return step_option


###############################################################################
# 4) HebbianTrainer - for purely feedforward Hebbian networks (no PCLayer)
###############################################################################
class HebbianTrainer:
    """
    For a purely feedforward network that might have HebbianLayers. 
    We do standard gradient-based training plus a Hebbian update each batch.
    """
    def __init__(
        self,
        model,
        optimizer_fn=optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
        do_hebbian_update=True
    ):
        self.model = model
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_kwargs)
        self.do_hebbian_update = do_hebbian_update

    def train_on_batch(self, data, loss_fn, loss_fn_kwargs=None):
        if loss_fn_kwargs is None:
            loss_fn_kwargs = {}
        # Forward
        out = self.model(data)
        loss = loss_fn(out, **loss_fn_kwargs)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # local Hebbian
        if self.do_hebbian_update:
            for m in self.model.modules():
                if hasattr(m, "hebbian_update"):
                    m.hebbian_update()

        return loss.item()


###############################################################################
# 5) Build Three Model Types (PC-Only, Hebb-Only, Hybrid)
###############################################################################
def build_model_pc_only(input_size, hidden_size, output_size):
    """
    PC-Only: uses nn.Linear + PCLayer, no HebbianLayer
    """
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        PCLayer(),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        PCLayer(),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )
    return model

def build_model_hebb_only(input_size, hidden_size, output_size):
    """
    Hebb-Only: purely feedforward net of HebbianLayers (no PCLayer)
    """
    model = nn.Sequential(
        HebbianLayer(input_size, hidden_size),
        nn.ReLU(),
        HebbianLayer(hidden_size, hidden_size),
        nn.ReLU(),
        HebbianLayer(hidden_size, output_size)
    )
    return model

def build_model_hybrid(input_size, hidden_size, output_size):
    """
    Hybrid: mixture of HebbianLayer + PCLayer
    """
    model = nn.Sequential(
        HebbianLayer(input_size, hidden_size),
        PCLayer(),
        nn.ReLU(),
        HebbianLayer(hidden_size, hidden_size),
        PCLayer(),
        nn.ReLU(),
        HebbianLayer(hidden_size, output_size)
    )
    return model


###############################################################################
# 6) Run experiments for each configuration
###############################################################################
def run_experiment_pc_only(
    train_loader, 
    test_dataset, 
    device,
    epochs=10,
    T=20,
    lr_x=0.01,
    lr_p=0.001,
    batch_size=500
):
    input_size = 28*28
    hidden_size = 256
    output_size = 10

    model = build_model_pc_only(input_size, hidden_size, output_size).to(device)
    trainer = PCTrainer(
        model=model,
        T=T,
        optimizer_x_fn=optim.SGD,
        optimizer_x_kwargs={"lr": lr_x},
        optimizer_p_fn=optim.Adam,
        optimizer_p_kwargs={"lr": lr_p},
        update_x_at="all",
        update_p_at="last",
        energy_coefficient=1.0,
        do_hebbian_update=False  # PC-only
    )

    # MSE loss with one-hot
    loss_fn = lambda out, _target: 0.5 * (out - _target).pow(2).sum()

    accs = []
    accs.append(test_accuracy(model, test_dataset, device))  # initial

    for ep in range(epochs):
        for data, label in tqdm.tqdm(train_loader, desc=f"[PC-Only] Epoch {ep+1}/{epochs}"):
            data, label = data.to(device), label.to(device)
            label_oh = F.one_hot(label, num_classes=output_size).float()
            energy_val, sup_loss_val = trainer.train_on_batch(
                data, loss_fn, loss_fn_kwargs={"_target": label_oh}
            )
        # Test
        acc = test_accuracy(model, test_dataset, device)
        accs.append(acc)
        print(f"  [PC-Only Epoch {ep+1}] Test Accuracy={acc:.4f}")

        ########################################################################
        # ADDITION: Print out weights (and latent x if needed) after each epoch
        ########################################################################
        # print("  [PC-Only] Model parameters after epoch", ep+1)
        # for name, param in model.named_parameters():
        #     print(f"    - Param '{name}' -> shape {param.data.shape}")
        #     print(param.data)
        # # Also print any PCLayers' x (latent states)
        # for pc in trainer.get_model_pc_layers():
        #     x = pc.get_x()
        #     if x is not None:
        #         print(f"    - PCLayer latent x -> shape {x.data.shape}")
        #         print(x.data)
        ########################################################################

    return accs


def run_experiment_hebb_only(
    train_loader, 
    test_dataset, 
    device,
    epochs=10,
    lr=0.001,
    batch_size=500
):
    input_size = 28*28
    hidden_size = 256
    output_size = 10

    model = build_model_hebb_only(input_size, hidden_size, output_size).to(device)
    trainer = HebbianTrainer(model=model, optimizer_fn=optim.Adam, optimizer_kwargs={"lr": lr}, do_hebbian_update=True)

    # MSE loss
    loss_fn = lambda out, _target: 0.5 * (out - _target).pow(2).sum()

    accs = []
    accs.append(test_accuracy(model, test_dataset, device))

    for ep in range(epochs):
        for data, label in tqdm.tqdm(train_loader, desc=f"[Hebb-Only] Epoch {ep+1}/{epochs}"):
            data, label = data.to(device), label.to(device)
            label_oh = F.one_hot(label, num_classes=output_size).float()
            loss_val = trainer.train_on_batch(data, loss_fn, loss_fn_kwargs={"_target": label_oh})
        acc = test_accuracy(model, test_dataset, device)
        accs.append(acc)
        print(f"  [Hebb-Only Epoch {ep+1}] Test Accuracy={acc:.4f}")

        ########################################################################
        # ADDITION: Print out weights after each epoch
        ########################################################################
        # print("  [Hebb-Only] Model parameters after epoch", ep+1)
        # for name, param in model.named_parameters():
        #     print(f"    - Param '{name}' -> shape {param.data.shape}")
        #     print(param.data)
        ########################################################################

    return accs


def run_experiment_hybrid(
    train_loader,
    test_dataset,
    device,
    epochs=10,
    T=20,
    lr_x=0.01,
    lr_p=0.001,
    batch_size=500
):
    input_size = 28*28
    hidden_size = 256
    output_size = 10

    model = build_model_hybrid(input_size, hidden_size, output_size).to(device)
    trainer = PCTrainer(
        model=model,
        T=T,
        optimizer_x_fn=optim.SGD,
        optimizer_x_kwargs={"lr": lr_x},
        optimizer_p_fn=optim.Adam,
        optimizer_p_kwargs={"lr": lr_p},
        update_x_at="all",
        update_p_at="last",
        energy_coefficient=1.0,
        do_hebbian_update=True  # hybrid => do Hebbian
    )

    loss_fn = lambda out, _target: 0.5 * (out - _target).pow(2).sum()

    accs = []
    accs.append(test_accuracy(model, test_dataset, device))

    for ep in range(epochs):
        for data, label in tqdm.tqdm(train_loader, desc=f"[Hybrid] Epoch {ep+1}/{epochs}"):
            data, label = data.to(device), label.to(device)
            label_oh = F.one_hot(label, num_classes=output_size).float()
            energy_val, sup_loss_val = trainer.train_on_batch(
                data, loss_fn, loss_fn_kwargs={"_target": label_oh}
            )
        acc = test_accuracy(model, test_dataset, device)
        accs.append(acc)
        print(f"  [Hybrid Epoch {ep+1}] Test Accuracy={acc:.4f}")

        ########################################################################
        # ADDITION: Print out weights (and latent x if needed) after each epoch
        ########################################################################
        # print("  [Hybrid] Model parameters after epoch", ep+1)
        # for name, param in model.named_parameters():
        #     print(f"    - Param '{name}' -> shape {param.data.shape}")
        #     print(param.data)
        # # Also print any PCLayers' x (latent states)
        # for pc in trainer.get_model_pc_layers():
        #     x = pc.get_x()
        #     if x is not None:
        #         print(f"    - PCLayer latent x -> shape {x.data.shape}")
        #         print(x.data)
        ########################################################################

    return accs


###############################################################################
# 7) test_accuracy helper
###############################################################################
def test_accuracy(model, dataset, device, batch_size=1000):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            _, pred = torch.max(out, dim=-1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


###############################################################################
# Add Noise to MNIST Dataset
###############################################################################
def add_noise_to_image(image, noise_std=1.0):
    """
    Add Gaussian noise to a single image.

    Args:
        image (torch.Tensor): The input image.
        noise_std (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Noisy image.
    """
    noise = torch.randn_like(image) * noise_std
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0.0, 1.0)  # Ensure values remain in [0, 1]


# Update transform to include noise for training data
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),  # Flatten
    transforms.Lambda(lambda x: add_noise_to_image(x, noise_std=0.5))  # Add noise
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten
])

# Load data with noise added to training set
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)

batch_size = 2048
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Seed
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters
    EPOCHS = 20
    batch_size = 2048
    noise_levels = [0.0, 0.2, 0.3, 0.4, 0.5]  # Different noise levels (0%, 20%, ..., 50%)

    # Results dictionary to store accuracies
    results = {
        "PC-Only": {},
        "Hebb-Only": {},
        "Hybrid": {}
    }

    for noise_std in noise_levels:
        print(f"\n### Training with Noise Level: {noise_std * 100:.0f}% ###")

        # Update transforms for the current noise level
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten
            transforms.Lambda(lambda x: add_noise_to_image(x, noise_std=noise_std))  # Add noise
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten
        ])

        # Load data
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Train and test PC-Only
        print("\n================= PC-Only =================")
        accs_pc_only = run_experiment_pc_only(
            train_loader, test_dataset, device,
            epochs=EPOCHS,
            T=20,
            lr_x=0.01,
            lr_p=0.001,
            batch_size=batch_size
        )
        results["PC-Only"][noise_std] = accs_pc_only

        # Train and test Hebb-Only
        print("\n================= Hebb-Only =================")
        accs_hebb_only = run_experiment_hebb_only(
            train_loader, test_dataset, device,
            epochs=EPOCHS,
            lr=0.001,
            batch_size=batch_size
        )
        results["Hebb-Only"][noise_std] = accs_hebb_only

        # Train and test Hybrid
        print("\n================= Hybrid =================")
        accs_hybrid = run_experiment_hybrid(
            train_loader, test_dataset, device,
            epochs=EPOCHS,
            T=20,
            lr_x=0.01,
            lr_p=0.001,
            batch_size=batch_size
        )
        results["Hybrid"][noise_std] = accs_hybrid

    # Plot results for all configurations and noise levels
    plt.figure(figsize=(10, 6))
    for config, config_results in results.items():
        for noise_std, accs in config_results.items():
            plt.plot(range(EPOCHS + 1), accs, label=f"{config} - Noise {noise_std * 100:.0f}%")
    plt.title("MNIST: Configurations with Varying Noise Levels")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save results to a text file
    output_file = "results_summary.txt"
    with open(output_file, "w") as f:
        f.write("MNIST: Configurations with Varying Noise Levels\n")
        f.write("="*50 + "\n")
        for config, config_results in results.items():
            f.write(f"\n### {config} ###\n")
            for noise_std, accs in config_results.items():
                f.write(f"Noise Level {noise_std * 100:.0f}%:\n")
                f.write(", ".join(f"{acc:.4f}" for acc in accs) + "\n")
        f.write("\nResults saved successfully!\n")
    
    print(f"Results have been saved to {output_file}.")


