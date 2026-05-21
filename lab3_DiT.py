import os
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math
import uuid
import random

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.func import vmap, jacrev
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from einops import rearrange
from einops.layers.torch import Rearrange

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: b d
        """
        pass

class LabeledSampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: b d
            - labels: b
        """
        pass

class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """
    def __init__(self, shape: List[int], std: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...; used to infer the device by self.dummy.device

    def sample(self, num_samples) -> torch.Tensor:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)
    
class GMM(nn.Module, LabeledSampleable):
  def __init__(self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor):
    super().__init__()
    self.means = nn.Buffer(means) # num_modes x data_dim
    self.covariances = nn.Buffer(covariances) # num_modes (0 correlation between data_dim)
    self.weights = nn.Buffer(weights) # num_modes; do not need to sum to 1 for torch.multinomial

  def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      - num_samples: the desired number of samples
    Returns:
      - samples: b n
      - labels: b
    """
    # Choose the label/mode of each sample; [b]
    # Perform multinomial sampling on CPU to avoid device-side assert errors
    labels = torch.multinomial(self.weights.cpu(), num_samples=num_samples, replacement=True).to(self.means.device)

    # Sample from each mode
    samples = torch.zeros(num_samples, self.means.shape[1]).to(self.means.device)
    for idx in range(len(self.means)):
      samples[labels == idx] = torch.randn_like(samples[labels == idx]) * self.covariances[idx] + self.means[idx]

    return samples, labels
  
class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """
    def __init__(self, p_simple: Sampleable, p_data: LabeledSampleable):
        super().__init__()
        self.p_simple = p_simple        # p_init
        self.p_data = p_data            # z ~ p_data
        # initial conditional prob.     p_0(x|z) = p_init(x)
        # endpoint conditional prob.    p_1(x|z)
        # endpoint marginal prob.       p_1(x) = int p_1(x|z) p_data(z) dz = p_data(x)

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = int p_t(x|z) p(z) dz
        Args:
            - t: b
        Returns:
            - x: samples from p_t(x), b ... (i.e.,. `b d`, `b c h w`, etc.)
        """
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p_data
        z, _ = self.p_data.sample(num_samples) # [b ...]
        # Sample conditional probability path x ~ p_t(x|z) (intermediate point along the prob. path)
        x = self.sample_conditional_path(z, t) # [b ...]
        return x

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable b ...
            - t: time b
        Returns:
            - x: samples from p_t(x|z), b ...
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: b ...
            - z: b ...
            - t: b
        Returns:
            - conditional_vector_field: conditional vector field [b c h w]
        """
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: b ...
            - z: b ...
            - t: b
        Returns:
            - score: b ...
        """
        pass

class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1,)), torch.zeros(1,)
        )
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1,)), torch.ones(1,)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: b
        Returns:
            - alpha_t: b
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: b
        Returns:
            - d/dt a_t: b
        """
        t = t.unsqueeze(1) # (b,) -> (b,1)
        # batched_derivative_fn = vmap(jacrev(self)) 
        # dt = batched_derivative_fn(t)        
        dt = vmap(jacrev(self))(t)
        return dt.view(-1)

class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1)), torch.ones(1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1)), torch.zeros(1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: b
        Returns:
            - beta_t: b
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: b
        Returns:
            - d/dt beta_t: b
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1)

class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: b
        Returns:
            - alpha_t: b
        """
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: b
        Returns:
            - d/dt alpha_t b
        """
        return torch.ones_like(t)

class LinearBeta(Beta):
    """
    Implements beta_t = 1-t
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: b
        Returns:
            - beta_t: b
        """
        return 1-t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: b
        Returns:
            - d/dt alpha_t: b
        """
        return - torch.ones_like(t)

class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    p_t(x|z) ~ N(alpha_t * z, beta_t^2 I)
    """

    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        p_simple = IsotropicGaussian(shape = p_simple_shape, std = 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
        # Rearrange creates an einops function that stores reshape pattern for broadcasting
        self.rearrange_scalar = Rearrange(f'b -> b{" 1" * len(p_simple_shape)}')

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: b ...
            - t: b
        Returns:
            - x: b ...
        """
        alpha_t = self.rearrange_scalar(self.alpha(t)) # [b 1 1 1]
        beta_t = self.rearrange_scalar(self.beta(t)) # [b 1 1 1]
        return alpha_t * z + beta_t * torch.randn_like(z)

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z); Eq. (20)
        Args:
            - x: b c h w
            - z: b c h w
            - t: b
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """
        alpha_t = self.rearrange_scalar(self.alpha(t)) # b
        beta_t = self.rearrange_scalar(self.beta(t)) # b
        dt_alpha_t = self.rearrange_scalar(self.alpha.dt(t)) # b
        dt_beta_t = self.rearrange_scalar(self.beta.dt(t)) # b

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z); Eq. (40)
        Args:
            - x: b ...
            - z: b ...
            - t: b
        Returns:
            - conditional_score: b ...
        """
        alpha_t = self.rearrange_scalar(self.alpha(t))
        beta_t = self.rearrange_scalar(self.beta(t))
        return (z * alpha_t - x) / beta_t ** 2
    
class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient u_t(xt) of the ODE.
        Args:
            - xt: b ...
            - t: b
        Returns:
            - drift_coefficient: b ...
        """
        pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient u_t(xt) of the SDE.
        Args:
            - xt: b ...
            - t: b
        Returns:
            - drift_coefficient: b ...
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the diffusion coefficient sigma_t of the SDE.
        Args:
            - xt: b ...
            - t: b
        Returns:
            - diffusion_coefficient: b ...
        """
        pass

class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs):
        """
        Takes one simulation step
        Args:
            - xt: [b ...]
            - t: [b]
            - dt: [b]
        Returns:
            - nxt: [b ...]
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, use_tqdm: bool = True, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: [b ...]
            - ts: [b nt]
        Returns:
            - x_final: [b ...]
        """
        nt = ts.shape[1]
        pbar = tqdm(range(nt - 1)) if use_tqdm else range(nt - 1)
        for t_idx in pbar:
            t = ts[:, t_idx]                    # [b]
            h = ts[:, t_idx + 1] - ts[:, t_idx] # [b]; dt
            x = self.step(x, t, h, **kwargs)    # [b]
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, use_tqdm: bool = True, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x: [b ...]
            - ts: [b nt]
        Returns:
            - x_traj: [b nt ...]
        """
        x_traj = [x.clone()]
        nt = ts.shape[1]
        pbar = tqdm(range(nt - 1)) if use_tqdm else range(nt - 1)
        for t_idx in pbar:
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            x_traj.append(x.clone())
        return torch.stack(x_traj, dim=1) # [b nt ...]; list to tensor

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        h = h.view([-1] + [1] * (len(xt.shape) - 1)) # h.shape == (b,) -> (b 1 1 1) for broadcasting
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h

class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        h = h.view([-1] + [1] * (len(xt.shape) - 1))
        # Eq. (9)
        return xt + self.sde.drift_coefficient(xt, t, **kwargs) * h + self.sde.diffusion_coefficient(xt, t, **kwargs) * torch.sqrt(h) * torch.randn_like(xt)

def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    Always records the first and the last time step (num_timesteps - 1)
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )

MiB = 1024 ** 2 # 1 MiB = 1024 × 1024 bytes

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(
        self,
        **kwargs
      ):
        super().__init__()      # Call super constructor if Trainer is subclassed by ABC
        self.model = None
        self.opt = None         # optimizer
        self.output_dir = None

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def checkpoint(self, step: int):
      pass

    def get_optimizer(self, lr: float):
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def random_name(self) -> str:
        adjectives = ["autumn", "hidden", "bitter", "misty", "silent", "empty", "dry", "dark", "summer", "icy", "delicate", "quiet", "white", "cool", "spring", "winter", "patient"]
        foods = ["apple", "banana", "pear", "plum", "orange", "persimmon", "tangerine", "durian", "jackfruit", "jicama", "cantaloupe", "watermelon", "peach"]
        return f"{random.choice(adjectives)}-{random.choice(foods)}-{str(uuid.uuid4())[:8]}"

    def train(
        self,
        model: nn.Module,
        num_steps: int,
        lr: float = 1e-3,
        warmup_steps: int = 500, # warmup training steps where the learning rate 0 -> lr
        ckpt_every: Optional[int] = 500,
        run_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[float], List[int]]:
        """
        Linear warmup from 0 -> lr over `warmup_steps`, then constant lr.
        """
        # Initialize run name and output directory
        run_name = run_name or self.random_name()
        self.output_dir = os.path.join("runs", run_name)
        os.makedirs(self.output_dir, exist_ok=False)
        print("Initialized output directory at: " + self.output_dir)

        # Grab size
        self.model = model
        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        # Initialize optimizer and LR
        self.opt = self.get_optimizer(lr)
        self.model.train() # puts the model into training mode
        
        # set optimizer's learning rate to 0
        for pg in self.opt.param_groups:
            pg["lr"] = 0.0

        # Main training loop
        losses: List[float] = []
        steps: List[int] = []

        pbar = tqdm(range(num_steps))
        for step in pbar:
            # Update LR
            if warmup_steps > 0 and step < warmup_steps:
                cur_lr = lr * float(step + 1) / float(warmup_steps)
            else:
                cur_lr = lr
            for pg in self.opt.param_groups:
                pg["lr"] = cur_lr

            # Forward + backward
            self.opt.zero_grad(set_to_none=True)
            loss = self.get_train_loss(**kwargs)
            loss.backward()

            # Take gradient step
            self.opt.step()

            losses.append(float(loss.detach().item()))
            steps.append(step)

            pbar.set_description(f"Step {step}, lr={cur_lr:.2e}, loss={loss.item():.4f}")

            # Callback if specified
            if ckpt_every is not None and step % ckpt_every == 0 and step > 0:
              self.model.eval()
              self.checkpoint(step)
              self.model.train()

        self.model.eval()
        return losses, steps
    
class MNISTSampler(nn.Module, LabeledSampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """
    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1305,), (0.2891,)), # x_normalized = (x - mean) / std; 1 channel, mean=0.1305, std=0.2891 computed from the MNIST training set
            ])
        )
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called; used as self.dummy.device

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples]               # randomly select num_samples
        samples, labels = zip(*[self.dataset[i] for i in indices])              # unzips that list of pairs
        samples = torch.stack(samples).to(self.dummy.device)                    # [batch_size, c, h, w]
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)  # [batch_size]
        return samples, labels
    
class ConditionalVectorField(nn.Module, ABC):
    """
    Conditional vector field u_t^theta(x|y)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: b ...
        - t: b
        - y: b
        Returns:
        - u_t^theta(x|y): b ...
        """
        pass

class CFGVectorFieldODE(ODE):
    def __init__(self, net: ConditionalVectorField, null_label: int, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale
        self.null_label = null_label

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b ...
        - t: b
        - y: b
        """
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.ones_like(y) * self.null_label
        unguided_vector_field = self.net(x, t, unguided_y)
        # Eq. (65): \tilde{u}_t^theta(x|y); [b ...]
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field

class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, eta: float, null_label: int, eps: float = 0.001, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(**kwargs)
        self.eta = eta # label dropout probability
        self.eps = eps # t in [0, 1 - eps); avoid beta_t = 0 at t=1
        self.path = path
        self.null_label = null_label

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size) # [b ...], [b]

        # Step 2: Set each label to 10 (i.e., null) with probability eta
        xi = torch.rand(y.shape[0]).to(y.device)
        y[xi < self.eta] = self.null_label

        # Step 3: Sample t and x
        t = torch.rand(batch_size).to(z) * (1 - self.eps) # [b]
        x = self.path.sample_conditional_path(z,t) # [b ...]

        # Step 4: Regress and output loss
        ut_theta = self.model(x,t,y) # [b ...]
        ut_ref = self.path.conditional_vector_field(x,z,t) # [b ...]
        return torch.square(ut_theta - ut_ref).mean()
    
# Sanity check: implement MLPConditionalVectorField

class MLP(nn.Module):
  def __init__(self, dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU, final_init: bool = False):
    super().__init__()
    mlp = []
    for idx in range(len(dims) - 1):
        mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            mlp.append(activation())
    self.net = torch.nn.Sequential(*mlp)

    if final_init:
      nn.init.zeros_(self.net[-1].weight)
      nn.init.zeros_(self.net[-1].bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    Returns:
    - x: b n d
    """
    return self.net(x)

class MLPConditionalVectorField(ConditionalVectorField):
  def __init__(
      self,
      dim: int,         # data dimension
      hidden_dim: int,  # 
      class_dim: int,   # dimension of class-label embedding
      num_classes: int  # 
    ):
    super().__init__()
    self.mlp = MLP([dim + class_dim + 1, hidden_dim, hidden_dim, dim]) # dim + class_dim + 1 = data + label + t
    # embed discrete class labels into a vector; the LU table is learnable; num_classes + 1 to account for null label
    self.class_embedding = nn.Embedding(num_classes + 1, class_dim)

  def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
      """
      Args:
      - x: b d
      - t: b
      - y: b
      Returns:
      - u_t^theta(x|y): [b ...]
      """
      xyt = torch.cat([
          x,
          self.class_embedding(y),
          t.unsqueeze(-1) # (1,) -> (b,1); unsqueeze for concatenation
      ], dim=-1)
      # xyt.shape == (b, dim + class_dim + 1)
      return self.mlp(xyt)

class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        # learnable w_i; initialized as random Gaussian
        self.weights = nn.Parameter(torch.randn(1, self.half_dim)) # [1 half_dim]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: [b]
        Returns:
        - embeddings: [b dim]
        """
        # Step 1: compute frequencies f_i = 2 * pi * w_i * t
        t = t.view(-1, 1) # [b] -> [b 1]
        freqs = t * self.weights * 2 * math.pi # [b half_dim]

        # Step 2: compute sin(f_i) and cos(f_i)
        sin_embed = torch.sin(freqs) # [b half_dim]
        cos_embed = torch.cos(freqs) # [b half_dim]

        # Step 3: Concatenate and return
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # [b dim]

class Patchifier(nn.Module):
  def __init__(self, img_h: int, img_w: int, patch_size: int, c_in: int, dim: int):
    super().__init__()
    assert img_h % patch_size == 0, "Image size must be divisible by patch size"
    assert img_w % patch_size == 0, "Image size must be divisible by patch size"

    self.net = nn.Sequential(
        # Initial convolution
        # kernel_size=stride=patch_size to create non-overlapping patches
        # each patch outputs dim (latent dimension) channels
        nn.Conv2d(c_in, dim, kernel_size=patch_size, stride=patch_size),

        # [b, dim, num_patches_h, num_patches_w] -> [b, num_patches, dim]
        # num_tokens = num_patches = (img_h / patch_size) * (img_w / patch_size)
        Rearrange("b d h w -> b (h w) d"),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: (b, c, h, w)
    Returns:
    - x: [b, num_patches, dim]
    """
    return self.net(x)

class MHA(nn.Module):
  """
  Multi-headed self-attention
  """
  def __init__(self, dim: int, heads: int):
    super().__init__()
    assert dim % heads == 0

    self.scale = (dim // heads) ** -0.5
    self.qkv = nn.Linear(dim, dim * 3)
    self.fold_heads = Rearrange('b n (h d) -> (b h) n d', h=heads)
    self.unfold_heads = Rearrange('(b h) n d -> b n (h d)', h=heads)
    self.out = nn.Linear(dim, dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    Returns:
    - x: b n d
    """
    # Compute queries, keys, and values
    q, k, v = self.qkv(x).chunk(3, dim=-1) # b n (h d)

    # Fold head into batch dimension
    q, k, v = map(self.fold_heads, (q, k, v)) # (b h) n d

    # Compute attention
    qk = torch.einsum('bid,bjd->bij', q, k) * self.scale # (b h) n n
    attn = torch.softmax(qk, dim=-1) # (b h) n n

    # Combine with values
    x = torch.einsum('bij,bjd->bid', attn, v) # (b h) n d

    # Unfold heads
    x = self.unfold_heads(x) # b n (h d)

    # Pass throuh FF and return
    return self.out(x)

def modulate(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    - scale: b n d
    - bias: b n d
    Returns:
    - x: b n d
    """
    return x * (1 + scale) + bias

class DiffusionTransformerLayer(nn.Module):
  def __init__(
      self,
      dim: int,
      heads: int,
  ):
    """
    Args:
    - n_tokens: sequence length (for sake of positional embeddings)
    - dim: dimension of hidden layers
    - heads: number of attention heads
    """
    super().__init__()

    # Normalization
    self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
    self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
    self.ada_ln = nn.Sequential(
        nn.RMSNorm(dim, elementwise_affine=False),
        nn.Linear(dim, dim * 6)
    )

    # Initialize conditioning to zero - stabilizes residual connection!
    nn.init.zeros_(self.ada_ln[1].weight)
    nn.init.zeros_(self.ada_ln[1].bias)

    # Attention
    self.attn = MHA(dim, heads)

    # Feedforward
    self.ff = MLP([dim, 4 * dim, dim])

  def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    - c: b d
    Returns:
    - x: b n d
    """
    # Compute conditioning gating, scaling, and bias
    c = rearrange(self.ada_ln(c), 'b d -> b 1 d') # b 1 d
    attn_scale, attn_bias, attn_gate, ff_scale, ff_bias, ff_gate = c.chunk(6, dim=-1)

    # Attention + FF
    x = x + attn_gate * self.attn(
      modulate(self.norm1(x), attn_scale, attn_bias)
    )
    x = x + ff_gate * self.ff(
      modulate(self.norm2(x), ff_scale, ff_bias)
    )
    return x

class DiffusionTransformer(nn.Module):
  def __init__(
      self,
      depth: int,
      n_tokens: int,
      dim: int,
      **layer_kwargs,
  ):
    """
    Args:
    - n_tokens: sequence length (for sake of positional embeddings)
    - dim: dimension of hidden layers
    - heads: number of attention heads
    - depth: number of layers
    """
    super().__init__()
    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(DiffusionTransformerLayer(dim=dim, **layer_kwargs))

    # Positional encodings
    self.pos_encodings = nn.Parameter(torch.randn(n_tokens, dim))

  def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    - c: b d
    Returns:
    - x: b n d
    """
    x = x + self.pos_encodings.unsqueeze(0)
    for layer in self.layers:
      x = layer(x, c)
    return x

class Depatchifier(nn.Module):
  def __init__(self, img_size: int, patch_size: int, dim: int, final_dim: int, c_out: int):
      super().__init__()
      self.patch_size = patch_size
      assert img_size % patch_size == 0, "Image size must be divisible by patch size"
      h = w = img_size // patch_size


      self.net = nn.Sequential(
          # Norm + MLP
          nn.RMSNorm(dim, elementwise_affine=False),
          MLP([dim, 4*dim, final_dim * patch_size ** 2]),

          # Depatchify
          Rearrange("b (h w) (f ph pw) -> b f (h ph) (w pw)", h=h, w=w, f=final_dim, ph=patch_size, pw=patch_size),

          # Final convolution
          nn.Conv2d(final_dim, c_out, kernel_size=3, padding=1)
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    Returns:
    - x: b 1 32 32
    """
    return self.net(x)
  
class DiffusionTransformerFlowModel(ConditionalVectorField):
  def __init__(
      self,
      img_size: int = 32,
      patch_size: int = 8,
      num_layers: int = 12,
      c: int = 1,
      dim: int = 256,
      heads: int = 4,
      final_dim: int = 10,
      n_classes: int = 11,
    ):
      super().__init__()
      # 0. Construct time_embedder and y_embedder
      self.time_embedder = FourierEncoder(dim)
      self.y_embedder = nn.Embedding(num_embeddings = n_classes, embedding_dim = dim)

      # 1. Construct patchifier
      self.patchifier = Patchifier(
          img_h=img_size,
          img_w=img_size,
          patch_size=patch_size,
          c_in=c,
          dim=dim
        )

      # 2. Construct DiT
      n_tokens = (img_size // patch_size) ** 2
      self.dit = DiffusionTransformer(
          depth=num_layers,
          n_tokens=n_tokens,
          dim=dim,
          heads=heads,
      )

      # 3. Construct de-patchifier
      self.depatchifier = Depatchifier(
          img_size=img_size,
          patch_size=patch_size,
          dim=dim,
          final_dim=final_dim,
          c_out=c
        )

  def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b 1 32 32
    - t: b 1 1 1
    - c: b 1 1 1
    Returns:
    - u_t^theta(x|y): b 1 32 32
    """
    # 1. Embed time and y
    t_embed = self.time_embedder(t) # b d
    y_embed = self.y_embedder(y) # b d

    # 2. Patchify
    x = self.patchifier(x) # b n d

    # 3. Pass through DiT
    x = self.dit(x, t_embed + y_embed) # b d

    # 4. Depatchify
    x = self.depatchifier(x) # b 1 32 32

    return x
  
##################
# Training utils #
##################

@torch.no_grad()
def visualize_output(model, path, samples_per_class: int = 10, num_timesteps: int = 100, guidance_scales: List[float] = [1.0, 3.0, 5.0], save_path: Optional[str] = None, use_tqdm: bool = True):
  # Graph
  fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))

  for idx, w in enumerate(guidance_scales):
      # Setup ode and simulator
      ode = CFGVectorFieldODE(model, guidance_scale=w, null_label=10)
      simulator = EulerSimulator(ode)

      # Sample initial conditions
      y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
      num_samples = y.shape[0]
      x0 = path.p_simple.sample(num_samples) # (num_samples, 1, 32, 32)

      # Simulate
      ts = torch.linspace(0,0.999,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
      x1 = simulator.simulate(x0, ts, y=y, use_tqdm=use_tqdm)

      # Plot
      v_min, v_max = x1.min(), x1.max()
      x1 = (x1 - v_min) / (v_max - v_min)
      grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(0,1))
      axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
      axes[idx].axis("off")
      axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)

  # Save
  if save_path is not None:
      plt.savefig(save_path)
      plt.close()
  else:
    plt.show()

class MNISTCFGTrainer(CFGTrainer):
  '''
  CFG Trainer with MNIST-specific callback
  '''
  def checkpoint(self, step: int):
    # Save model
    torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'step_{step:6d}_model.pt'))
    torch.save(self.opt.state_dict(), os.path.join(self.output_dir, f'step_{step:6d}_opt.pt'))

    # Save output visualization
    visualize_output(self.model, self.path, save_path=os.path.join(self.output_dir, f'step_{step:6d}_output.png'), use_tqdm=False)

#################
# Training code #
#################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data = MNISTSampler(),
    p_simple_shape = [1, 32, 32],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# Initialize model
dit = DiffusionTransformerFlowModel(
    img_size = 32,
    patch_size = 4,
    num_layers = 8,
    c = 1,
    dim = 256,
    heads = 8,
    final_dim = 10,
    n_classes = 11,
).to(device)

# Initialize trainer
trainer = MNISTCFGTrainer(path = path, eta=0.35, null_label=10)

# Train! You should have reasonable results in ~15 A100 minutes
losses, steps = trainer.train(model=dit, num_steps = 20000, lr=0.4e-3, batch_size=256, ckpt_every=1000)

plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss vs. Step")
plt.show()

# Play with these!
samples_per_class = 10
num_timesteps = 100
guidance_scales = [1.0, 3.0, 5.0]

visualize_output(
    model=dit,
    path=path,
    samples_per_class=samples_per_class,
    num_timesteps=num_timesteps,
    guidance_scales=guidance_scales,
)
plt.show()