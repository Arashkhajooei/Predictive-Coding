import typing
import warnings
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# ----------------------------
# Predictive Coding Library
# ----------------------------

class PCLayer(nn.Module):
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
        self._energy_fn, self._energy_fn_kwargs, self._sample_x_fn = energy_fn, energy_fn_kwargs, sample_x_fn
        self.is_holding_error, self.is_keep_energy_per_datapoint = is_holding_error, is_keep_energy_per_datapoint
        self._S, self._M, self._is_sample_x, self._x = S, M, False, None
        self.clear_energy()
        if self.is_keep_energy_per_datapoint: self.clear_energy_per_datapoint()
        self.eval()

    def set_is_sample_x(self, is_sample_x): self._is_sample_x = is_sample_x
    def get_x(self): return self._x

    def energy(self): return self._energy
    def clear_energy(self): self._energy = None

    def energy_per_datapoint(self):
        assert self.is_keep_energy_per_datapoint
        return self._energy_per_datapoint

    def clear_energy_per_datapoint(self):
        if self.is_keep_energy_per_datapoint: self._energy_per_datapoint = None

    def forward(self, mu, energy_fn_additional_inputs={}):
        if not self.training:
            return mu
    
        # Check and initialize _x if needed
        if not self._is_sample_x:
            self._is_sample_x = (
                self._x is None or mu.device != self._x.device or mu.size() != self._x.size()
            )
    
        if self._is_sample_x:
            self._x = nn.Parameter(
                self._sample_x_fn({'mu': mu, 'x': self._x}).to(mu.device), requires_grad=True
            )
            self._is_sample_x = False
    
        x, energy_fn_inputs = self._x, {**{'mu': mu, 'x': self._x}, **energy_fn_additional_inputs}
    
        # Expand dimensions for structured energy computation if _S is defined
        if self._S is not None:
            assert mu.dim() == 2 and x.dim() == 2 and self._S.size() == (mu.size(1), x.size(1))
            mu, x = mu.unsqueeze(2).expand(-1, -1, x.size(1)), x.unsqueeze(1).expand(-1, mu.size(1), -1)
    
        # Compute energy
        energy = self._energy_fn(energy_fn_inputs, **self._energy_fn_kwargs)
        if self._S is not None:
            energy *= self._S.unsqueeze(0)
        elif self._M is not None:
            energy *= self._M.unsqueeze(0)
    
        # Keep energy per datapoint if required
        if self.is_keep_energy_per_datapoint:
            self._energy_per_datapoint = energy.sum(dim=list(range(1, energy.dim()))).unsqueeze(1)
    
        # Update energy and error
        self._energy = energy.sum()
        if self.is_holding_error:
            self.error = (self._x.data - mu).detach().clone()
    
        return self._x




class PCTrainer(object):
    """A trainer for predictive-coding models implemented using PCLayers."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_x_fn: typing.Callable = optim.SGD,
        optimizer_x_kwargs: dict = {"lr": 0.1},
        manual_optimizer_x_fn: typing.Callable = None,
        x_lr_amplifier: float = 1.0,
        x_lr_discount: float = 1.0,
        loss_x_fn: typing.Callable = None,
        loss_inputs_fn: typing.Callable = None,
        optimizer_p_fn: typing.Callable = optim.Adam,
        optimizer_p_kwargs: dict = {"lr": 0.001},
        manual_optimizer_p_fn: typing.Callable = None,
        T: int = 512,
        update_x_at: typing.Union[str, typing.List[int]] = "all",
        update_p_at: typing.Union[str, typing.List[int]] = "last",
        accumulate_p_at: typing.Union[str, typing.List[int]] = "never",
        energy_coefficient: float = 1.0,
        early_stop_condition: str = "False",
        update_p_at_early_stop: bool = True,
        plot_progress_at: typing.Union[str, typing.List[int]] = [],
        is_disable_warning_energy_from_different_batch_sizes: bool = False,
    ):
        """Initialize PCTrainer with model and training configurations."""
        self._model = model
        self._optimizer_x_fn = optimizer_x_fn
        self._optimizer_x_kwargs = optimizer_x_kwargs
        self._manual_optimizer_x_fn = manual_optimizer_x_fn
        self._optimizer_x = None

        self._x_lr_discount = x_lr_discount
        self._x_lr_amplifier = x_lr_amplifier

        self._loss_x_fn = loss_x_fn
        self._loss_inputs_fn = loss_inputs_fn

        self._optimizer_p_fn = optimizer_p_fn
        self._optimizer_p_kwargs = optimizer_p_kwargs
        self._manual_optimizer_p_fn = manual_optimizer_p_fn

        self.recreate_optimize_p()

        self._T = T

        self._update_x_at = self._preprocess_step_index_list(update_x_at, self._T)
        self._update_p_at = self._preprocess_step_index_list(update_p_at, self._T)
        self._accumulate_p_at = self._preprocess_step_index_list(accumulate_p_at, self._T)
        self._energy_coefficient = energy_coefficient
        self._early_stop_condition = early_stop_condition
        self._update_p_at_early_stop = update_p_at_early_stop
        self.is_disable_warning_energy_from_different_batch_sizes = is_disable_warning_energy_from_different_batch_sizes

    # GETTERS & SETTERS


    def get_model(self) -> nn.Module:
        return self._model

    def set_optimizer_x(self, optimizer_x: optim.Optimizer) -> None:
        assert isinstance(optimizer_x, optim.Optimizer)
        self._optimizer_x = optimizer_x

    def set_optimizer_x_lr(self, lr: float) -> None:
        for param_group in self._optimizer_x.param_groups:
            param_group['lr'] = lr

    def get_optimizer_p(self) -> optim.Optimizer:
        return self._optimizer_p

    def set_optimizer_p(self, optimizer_p: optim.Optimizer) -> None:
        assert isinstance(optimizer_p, optim.Optimizer)
        self._optimizer_p = optimizer_p

    def get_is_model_has_pc_layers(self) -> bool:
        """Check if the model contains any PCLayers."""
        for _ in self.get_model_pc_layers():
            return True
        return False

    def get_model_pc_layers(self) -> typing.Generator[PCLayer, None, None]:
        """Retrieve all PCLayers in the model."""
        for module in self._model.modules():
            if isinstance(module, PCLayer):
                yield module

    def get_named_model_pc_layers(self) -> typing.Generator[typing.Tuple[str, PCLayer], None, None]:
        """Retrieve all PCLayers in the model with their names."""
        for name, module in self._model.named_modules():
            if isinstance(module, PCLayer):
                yield name, module

    def get_model_xs(self, is_warning_x_not_initialized=True) -> typing.Generator[nn.Parameter, None, None]:
        """Retrieve all x parameters from PCLayers."""
        for pc_layer in self.get_model_pc_layers():
            model_x = pc_layer.get_x()
            if model_x is not None:
                yield model_x
            else:
                if is_warning_x_not_initialized:
                    warnings.warn(
                        "Some PCLayers have not been initialized (x is None).",
                        category=RuntimeWarning
                    )

    def get_model_parameters(self) -> typing.Generator[nn.Parameter, None, None]:
        """Retrieve trainable parameters excluding x parameters."""
        all_model_xs = set(self.get_model_xs(is_warning_x_not_initialized=False))
        for param in self._model.parameters():
            if not any(param is x for x in all_model_xs):
                yield param

    def get_numparameters(self, is_gen=True):
        parameters = self.get_model_parameters()
        if is_gen:
            num_params = sum((p.numel() if i != 0 else 0) for i, p in enumerate(parameters))
        else:
            num_params = sum(p.numel() for p in parameters)
        return num_params

    def get_weights_norms(self):
        weights_abs = []
        mu_abs = []
        params = self.get_model_parameters()
        for par in params:
            if len(par.size()) == 1:
                mu_abs.append(par.abs().mean())
            elif len(par.size()) == 2:
                weights_abs.append(par.abs().mean())
        return weights_abs, mu_abs

    def get_model_representations(self):
        xs = self.get_model_xs()
        for x in xs:
            return x.clone().detach().cpu()

    def get_model_xs_copy(self):
        xs = self.get_model_xs()
        out = []
        for x in xs:
            out.append(x.clone().detach().cpu())
        return out

    def get_num_pc_layers(self) -> int:
        """Get the number of PCLayers in the model."""
        return sum(1 for _ in self.get_model_pc_layers())

    def get_least_T(self) -> int:
        """Get the minimum required T based on the number of PCLayers."""
        return self.get_num_pc_layers() + 1

    # METHODS

    def recreate_optimize_x(self) -> None:
        """Recreate the optimizer for x."""
        if self._manual_optimizer_x_fn is None:
            self._optimizer_x = self._optimizer_x_fn(
                self.get_model_xs(),
                **self._optimizer_x_kwargs
            )
        else:
            self._optimizer_x = self._manual_optimizer_x_fn()

    def recreate_optimize_p(self) -> None:
        """Recreate the optimizer for model parameters."""
        if self._manual_optimizer_p_fn is None:
            self._optimizer_p = self._optimizer_p_fn(
                self.get_model_parameters(),
                **self._optimizer_p_kwargs
            )
        else:
            self._optimizer_p = self._manual_optimizer_p_fn()

        
    def train_on_batch(
        self,
        inputs,
        loss_fn=None,
        loss_fn_kwargs=None,
    ):
        """
        Train the model on a single batch with configurable loss function.
        """
        # Default configurations
        loss_fn = loss_fn or self.default_loss_fn
        loss_fn_kwargs = loss_fn_kwargs or {}
        is_sample_x_at_batch_start = True
        is_reset_optimizer_x_at_batch_start = True
        is_reset_optimizer_p_at_batch_start = False
        is_unwrap_inputs = False
        is_optimize_inputs = False
        is_clear_energy_after_use = False
        is_return_outputs = False
        is_return_representations = False
        is_return_xs = False
        is_return_batchelement_loss = False
    
        t_iterator = tqdm.trange(self._T) if False else range(self._T)
    
        results = {
            "loss": [],
            "energy": [],
            "overall": [],
            "outputs": [] if is_return_outputs else None,
            "representations": [] if is_return_representations else None,
            "xs": [] if is_return_xs else None,
        }
    
        is_dynamic_x_lr = self._x_lr_discount < 1.0 or self._x_lr_amplifier > 1.0
        overalls = [] if is_dynamic_x_lr else None
    
        unwrap_with = "**" if is_unwrap_inputs and isinstance(inputs, dict) else "*" if isinstance(inputs, (list, tuple)) else ""
    
        lr = [1.0] * 50
    
        for t in t_iterator:
            if t == 0 and self.get_is_model_has_pc_layers():
                if is_sample_x_at_batch_start:
                    for pc_layer in self.get_model_pc_layers():
                        pc_layer.set_is_sample_x(True)
                if is_optimize_inputs:
                    self.inputs = torch.nn.Parameter(self.inputs, requires_grad=True)
    
            outputs = (
                self._model(inputs).clone() if unwrap_with == "" else
                self._model(*inputs).clone() if unwrap_with == "*" else
                self._model(**inputs).clone()
            )
    
            if t == 0 and self.get_is_model_has_pc_layers():
                if is_sample_x_at_batch_start or is_reset_optimizer_x_at_batch_start:
                    self.recreate_optimize_x()
                if is_optimize_inputs:
                    self._optimizer_x.param_groups[0]["params"].append(self.inputs)
                model_xs = list(self.get_model_xs())
                if is_reset_optimizer_p_at_batch_start:
                    self.recreate_optimize_p()
    
            if is_return_outputs:
                results["outputs"].append(outputs)
            if is_return_representations:
                results["representations"].append(self.get_model_representations().clone().detach().cpu())
            if is_return_xs:
                results["xs"].append(self.get_model_xs_copy())
    
            loss = loss_fn(outputs, **loss_fn_kwargs) if loss_fn else None
            if loss:
                results["loss"].append(loss.item())
    
            energy = sum(self.get_energies(is_per_datapoint=False)) if self.get_is_model_has_pc_layers() else None
            if energy and is_clear_energy_after_use:
                for pc_layer in self.get_model_pc_layers():
                    pc_layer.clear_energy()
            if energy:
                results["energy"].append(energy.item())
    
            loss_x = sum([self._loss_x_fn(model_x) for model_x in model_xs]).sum() if self._loss_x_fn else None
            loss_inputs = self._loss_inputs_fn(self.inputs) if is_optimize_inputs and self._loss_inputs_fn else None
    
            overall = sum(filter(None, [
                loss,
                energy * self._energy_coefficient if energy else None,
                loss_x,
                loss_inputs
            ]))
    
            if is_dynamic_x_lr:
                overalls.append(overall)
    
            results["overall"].append(overall.item())
            if is_return_batchelement_loss:
                loss_kwargs_tmp = {**loss_fn_kwargs, "_reduction": "none"}
                energies_elem = sum(self.get_energies(is_per_datapoint=True)).squeeze()
                loss_elem = loss_fn(outputs, **loss_kwargs_tmp).sum(-1)
                results["overall_elementwise"] = energies_elem + loss_elem
    
            early_stop = eval(self._early_stop_condition)
    
            if self.get_is_model_has_pc_layers():
                if t in self._update_x_at:
                    self._optimizer_x.zero_grad()
    
            if ((t in self._update_p_at) or (early_stop and self._update_p_at_early_stop)) and (t not in self._accumulate_p_at):
                self._optimizer_p.zero_grad()
    
            if self._accumulate_p_at and t == self._accumulate_p_at[0]:
                self._optimizer_p.zero_grad()
    
            overall.backward()
    
            if self.get_is_model_has_pc_layers():
                lr.append(self._optimizer_x.param_groups[0]['lr'])
                lr.pop(0)
                if t in self._update_x_at:
                    self._optimizer_x.step()
                    if is_dynamic_x_lr and len(overalls) >= 2:
                        adjustment = self._x_lr_discount if overalls[-1] >= overalls[-2] else self._x_lr_amplifier
                        for param_group in self._optimizer_x.param_groups:
                            param_group['lr'] *= adjustment
    
            if t in self._update_p_at or (early_stop and self._update_p_at_early_stop):
                if self._accumulate_p_at:
                    for param in self.get_model_parameters():
                        param.grad /= len(self._accumulate_p_at)
                self._optimizer_p.step()
    
            if early_stop:
                break
    
        return results


    

    
    def _preprocess_step_index_list(self, indices: typing.Union[str, typing.List[int]], T: int) -> typing.List[int]:
        """Convert step indices from string or list to a list of integers."""
        if isinstance(indices, str):
            return {
                "all": list(range(T)),
                "last": [T - 1],
                "last_half": list(range(T // 2, T)),
                "never": []
            }.get(indices, [])
        return indices



    def get_model_pc_layers_training(self) -> list:
        """Get training status of all PCLayers."""
        return [pc_layer.training for pc_layer in self.get_model_pc_layers()]

    def get_energies(self, is_per_datapoint: bool = False, named_layers: bool = False) -> typing.Union[list, typing.Dict[str, PCLayer]]:
        """Retrieve energies from all PCLayers."""
        energies = {
            name: (pc_layer.energy_per_datapoint() if is_per_datapoint else pc_layer.energy())
            for name, pc_layer in self.get_named_model_pc_layers()
            if pc_layer.energy() is not None
        }

    
        return energies if named_layers else list(energies.values())

# ----------------------------
# MNIST Supervised Learning Example
# ----------------------------

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x))
])

train_dataset = datasets.MNIST('./data', download=True, train=True, transform=transform)
test_dataset = datasets.MNIST('./data', download=True, train=False, transform=transform)

batch_size = 500
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f'# Train images: {len(train_dataset)} | # Test images: {len(test_dataset)}')

# Define the model
input_size = 28 * 28  # MNIST images are 28x28
hidden_size = 256
output_size = 10    # 10 classes for digits 0-9
activation_fn = nn.ReLU

# Define the loss function
loss_fn = lambda output, _target: 0.5 * (output - _target).pow(2).sum()

model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    PCLayer(),
    activation_fn(),
    nn.Linear(hidden_size, hidden_size),
    PCLayer(),
    activation_fn(),
    nn.Linear(hidden_size, output_size)
)

model.train()   # Set the model to training mode
model.to(device)
print(model)

# Define the trainer
T = 20  # Number of inference iterations

optimizer_x_fn = optim.SGD          # Optimizer for latent state x
optimizer_x_kwargs = {'lr': 0.01}  # Optimizer parameters for x

update_p_at = 'last'                # Update parameters p at the last iteration
optimizer_p_fn = optim.Adam         # Optimizer for parameters p
optimizer_p_kwargs = {'lr': 0.001}  # Optimizer parameters for p

trainer = PCTrainer(
    model,
    T=T,
    optimizer_x_fn=optimizer_x_fn,
    optimizer_x_kwargs=optimizer_x_kwargs,
    update_p_at=update_p_at,
    optimizer_p_fn=optimizer_p_fn,
    optimizer_p_kwargs=optimizer_p_kwargs,
)

# Define the testing function
def test(model, dataset, batch_size=1000):
    model.eval()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            _, predicted = torch.max(pred, -1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    model.train()
    return round(correct / total, 4)

# Training loop with energy tracking
# Training loop with energy tracking
epochs = 10
test_acc = np.zeros(epochs + 1)
test_acc[0] = test(model, test_dataset)
epoch_energies = []  # To store energy values for each epoch

print(f'Initial Test Accuracy: {test_acc[0]}')

for epoch in range(epochs):
    epoch_energy = 0  # Reset energy accumulator for the epoch
    batch_count = 0   # Count batches to compute average energy per epoch

    # Initialize the tqdm progress bar
    with tqdm(train_loader, desc=f'Epoch {epoch+1} - Test accuracy: {test_acc[epoch]:.4f}') as pbar:
        for data, label in pbar:
            data, label = data.to(device), label.to(device)
            # Convert labels to one-hot encoding
            label = F.one_hot(label, num_classes=output_size).float()
            # Train on the current batch
            results = trainer.train_on_batch(
                inputs=data,
                loss_fn=loss_fn,
                loss_fn_kwargs={'_target': label},
            )
            # Get the energy from the final step
            batch_energy = sum(trainer.get_energies())  # Sum the energies across all layers
            epoch_energy += batch_energy.item()
            batch_count += 1

    # Compute average energy for the epoch
    average_energy = epoch_energy / batch_count
    epoch_energies.append(average_energy)

    # Evaluate on the test set
    test_acc[epoch + 1] = test(model, test_dataset)
    pbar.set_description(f'Epoch {epoch + 1} - Test accuracy: {test_acc[epoch + 1]:.4f} - Avg energy: {average_energy:.4f}')
    print(f"Epoch {epoch + 1} completed. Test Accuracy: {test_acc[epoch + 1]:.4f}, Average Energy: {average_energy:.4f}")

# Plot test accuracy and energy over epochs
plt.figure(figsize=(12, 6))
plt.plot(range(epochs + 1), test_acc, marker='o', label='Test Accuracy')
plt.plot(range(1, epochs + 1), epoch_energies, marker='s', label='Average Energy', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Test Accuracy and Average Energy over Epochs')
plt.grid(True)
plt.legend()
plt.show()

