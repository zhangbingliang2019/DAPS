from torch.autograd import grad
from abc import ABC, abstractmethod
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
import tqdm
import torch
import warnings

__DIFFUSION_SCHEDULER__ = {}


def register_diffusion_scheduler(name: str):
    def wrapper(cls):
        if __DIFFUSION_SCHEDULER__.get(name, None):
            if __DIFFUSION_SCHEDULER__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __DIFFUSION_SCHEDULER__[name] = cls
        cls.name = name
        return cls

    return wrapper


def get_diffusion_scheduler(name: str, **kwargs):
    if __DIFFUSION_SCHEDULER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DIFFUSION_SCHEDULER__[name](**kwargs)


class Scheduler(ABC):
    """
    Abstract base class for diffusion scheduler.

    Schedulers manage time steps, noise scales (sigma), scaling factors, and coefficients 
    used in diffusion stochastic/ordinary differential equations (SDEs/ODEs).
    """

    def __init__(self, num_steps):
        self.num_steps = num_steps + 1 # include the initial step

    def discretize(self, time_steps):
        sigma_steps = self.get_sigma(time_steps[:-1])
        sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])
        self.sigma_steps = sigma_steps

    def tensorize(self, data):
        if isinstance(data, (int, float)):
            return torch.tensor(data).float()
        if isinstance(data, list):
            return torch.tensor(data).float()
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        if isinstance(data, torch.Tensor):
            return data.float()
        raise ValueError(f"Data type {type(data)} is not supported.") 

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Noise Scheduling & Scaling Function 
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @abstractmethod
    def get_scaling(self, t):
        pass
    
    def get_sigma(self, t):
        pass
    
    def get_scaling_derivative(self, t):
        pass

    def get_sigma_derivative(self, t):
        pass

    def get_sigma_inv(self, sigma):
        pass
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Time & Sigma Range Function
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_t_min(self):
        pass

    def get_t_max(self):
        pass

    def get_discrete_time_steps(self, num_steps):
        pass

    def get_sigma_max(self):
        return self.get_sigma(self.get_t_max())

    def get_sigma_min(self):
        return self.get_sigma(self.get_t_min())
    
    def get_prior_sigma(self):
        # simga(t_max) * scaling(t_max)
        return self.get_sigma_max() * self.get_scaling(self.get_t_max())

    def summary(self):
        print('+' * 50)
        print('Diffusion Scheduler Summary')
        print('+' * 50)
        print(f"Scheduler       : {self.name}")
        print(f"Time Range      : [{self.get_t_min().item()}, {self.get_t_max().item()}]")
        print(f"Sigma Range     : [{self.get_sigma_min().item()}, {self.get_sigma_max().item()}]")
        print(f"Scaling Range   : [{self.get_scaling(self.get_t_min()).item()}, {self.get_scaling(self.get_t_max()).item()}]")
        print(f"Prior Sigma     : {self.get_prior_sigma().item()}")
        print('+' * 50)
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # For Iterating Over the Discretized Scheduler
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __iter__(self):
        self.pbar = tqdm.trange(self.num_steps) if self.verbose else range(self.num_steps)
        self.pbar_iter = iter(self.pbar)
        return self

    def __next__(self):
        try:
            step = next(self.pbar_iter)
            time, scaling, sigma, scaling_factor, factor = self.time_steps[step], self.scaling_steps[step], \
                self.sigma_steps[step], self.scaling_factor_steps[step], self.factor_steps[step]
            return self.pbar, time, scaling, sigma, factor, scaling_factor
        except StopIteration:
            raise StopIteration


@register_diffusion_scheduler('vp')
class VPScheduler(Scheduler):
    """Variance Preserving Scheduler."""

    def __init__(self, num_steps, beta_max=20, beta_min=0.1, epsilon=1e-5, beta_type='linear'):
        super().__init__(num_steps)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_type = beta_type
        self.epsilon = epsilon

        if beta_type == 'linear':
            self.n = 1
        elif beta_type == 'scaled_linear':
            self.n = 2
        else:
            raise NotImplementedError
        
        self.a = beta_max ** (1 / self.n) - beta_min ** (1 / self.n)
        self.b = beta_min ** (1 / self.n)

        time_steps = self.get_discrete_time_steps(self.num_steps)
        self.discretize(time_steps)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # For VP Scheduler Only
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_beta(self, t):
        # beta(t) = (a * t + b) ^ n
        t = self.tensorize(t)
        return (self.a * t + self.b) ** self.n

    def get_beta_integrated(self, t):
        # beta_integrated(t) = [(a * t + b) ^ (n + 1) - b ^ (n + 1)] / a / (n + 1)
        t = self.tensorize(t)
        return ((self.a * t + self.b) ** (self.n + 1) - self.b ** (self.n + 1)) / self.a / (self.n + 1)

    def get_alpha(self, t):
        # alpha(t) = exp(-beta_integrated(t))
        t = self.tensorize(t)
        return torch.exp(-self.get_beta_integrated(t))

    def get_alpha_derivative(self, t):
        # alpha'(t) = -beta(t) * alpha(t)
        t = self.tensorize(t)
        return - self.get_beta(t) * self.get_alpha(t)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # General Interface
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_scaling(self, t):
        # s(t) = sqrt(alpha(t))
        t = self.tensorize(t)
        return torch.sqrt(self.get_alpha(t))

    def get_sigma(self, t):
        # sigma(t) = sqrt(1 / alpha(t) - 1)
        t = self.tensorize(t)
        return torch.sqrt(1 / self.get_alpha(t) - 1)

    def get_scaling_derivative(self, t):
        # s'(t) = -s(t) * beta(t) / 2
        t = self.tensorize(t)
        return - self.get_scaling(t) * self.get_beta(t) / 2

    def get_sigma_derivative(self, t):
        # sigma'(t) = beta(t) / 2 / sigma(t) / alpha(t)
        t = self.tensorize(t)
        return self.get_beta(t) / 2 / self.get_sigma(t) / self.get_alpha(t)

    def get_sigma_inv(self, sigma):
        # t = {[a(n+1)log(sigma^2 + 1) + b^(n+1)]^(1/(n + 1)) - b}/a
        sigma = self.tensorize(sigma)
        return ((self.a * (self.n + 1) * torch.log(sigma ** 2 + 1) + self.b ** (self.n + 1)) ** (1 / (self.n + 1)) - self.b) / self.a

    def get_t_min(self):
        return self.tensorize(self.epsilon)
    
    def get_t_max(self):
        return self.tensorize(1)

    def get_discrete_time_steps(self, num_steps):
        return torch.linspace(1, self.epsilon, num_steps)


@register_diffusion_scheduler('ve')
class VEScheduler(Scheduler):
    """Variance Exploding Scheduler."""

    def __init__(self, num_steps, sigma_max=100, sigma_min=1e-2):
        super().__init__(num_steps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # get time_steps
        time_steps = self.get_discrete_time_steps(self.num_steps)
        self.discretize(time_steps)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # General Interface
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_sigma(self, t):
        # sigma(t) = sqrt(t)
        t = self.tensorize(t)
        return t.sqrt()

    def get_scaling(self, t):
        # s(t) = 1
        t = self.tensorize(t)
        return torch.ones_like(t)

    def get_sigma_derivative(self, t):
        # sigma'(t) = 1 / 2 / sqrt(t)
        t = self.tensorize(t)
        return 1 / t.sqrt() / 2

    def get_scaling_derivative(self, t):
        # s'(t) = 0
        t = self.tensorize(t)
        return torch.zeros_like(t)

    def get_sigma_inv(self, sigma):
        # t = sigma^2
        sigma = self.tensorize(sigma)
        return sigma ** 2

    def get_t_min(self):
        return self.tensorize(self.sigma_min ** 2)
    
    def get_t_max(self):
        return self.tensorize(self.sigma_max ** 2)

    def get_discrete_time_steps(self, num_steps):
        time_steps_fn = lambda r: self.sigma_max ** 2 * (self.sigma_min ** 2 / self.sigma_max ** 2) ** r
        steps = np.linspace(0, 1, num_steps)
        time_steps = np.array([time_steps_fn(s) for s in steps])
        return torch.from_numpy(time_steps).float()


@register_diffusion_scheduler('edm')
class EDMScheduler(Scheduler):
    """
        EDM (Elucidating the Design Space of Diffusion-Based Generative Models) Scheduler.
    """

    def __init__(self, num_steps, sigma_max=100, sigma_min=1e-2, timestep='poly-7'):
        super().__init__(num_steps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        p = int(timestep.split('-')[1])
        self.time_steps_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p

        # get time_steps
        time_steps = self.get_discrete_time_steps(self.num_steps)
        self.discretize(time_steps)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # General Interface
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_sigma(self, t):
        # sigma(t) = t
        return self.tensorize(t)

    def get_scaling(self, t):
        # s(t) = 1
        return torch.ones_like(self.tensorize(t))

    def get_sigma_derivative(self, t):
        # sigma'(t) = 1
        return torch.ones_like(self.tensorize(t))

    def get_scaling_derivative(self, t):
        # s'(t) = 0
        return torch.zeros_like(self.tensorize(t))
    
    def get_sigma_inv(self, sigma):
        return self.tensorize(sigma)

    def get_t_min(self):
        return self.tensorize(self.sigma_min)
    
    def get_t_max(self):
        return self.tensorize(self.sigma_max)

    def get_discrete_time_steps(self, num_steps):
        steps = np.linspace(0, 1, num_steps)
        time_steps = np.array([self.time_steps_fn(s) for s in steps])
        return torch.from_numpy(time_steps)
    

@register_diffusion_scheduler('trigflow')
class TrigFlowScheduler(Scheduler):
    """TrigFlow (Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models) Scheduler."""
    def __init__(self, num_steps, sigma_d=1.0, sigma_max=100, sigma_min=1e-2):
        super().__init__(num_steps)
        self.sigma_d = sigma_d
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min 

        # get time_steps
        time_steps = self.get_discrete_time_steps(self.num_steps)
        self.discretize(time_steps)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # General Interface
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_sigma(self, t):
        # sigma(t) = tan(t)
        return torch.tan(self.tensorize(t))

    def get_scaling(self, t):
        # s(t) = cos(t)
        return torch.cos(self.tensorize(t))

    def get_sigma_derivative(self, t):
        # sigma'(t) = 1 / cos^2(t)
        return 1 / torch.cos(self.tensorize(t)) ** 2

    def get_scaling_derivative(self, t):
        # s'(t) = -sin(t)
        return -torch.sin(self.tensorize(t))
    
    def get_sigma_inv(self, sigma):
        return torch.arctan(self.tensorize(sigma))

    def get_t_min(self):
        return self.get_sigma_inv(self.sigma_min)
    
    def get_t_max(self):
        return self.get_sigma_inv(self.sigma_max)

    def get_prior_sigma(self):
        return super().get_prior_sigma() * self.sigma_d
    
    def get_discrete_time_steps(self, num_steps):
        return torch.linspace(self.get_t_max().item(), self.get_t_min().item(), num_steps)


class DiffusionPFODE:
    """
    Diffusion Probability Flow ODE (PF-ODE) for sampling and likelihood computation.

    Implements forward and reverse sampling based on diffusion models, using numerical ODE solvers.
    """
    def __init__(self, model, scheduler, solver='euler'):
        self.model = model
        self.scheduler = scheduler
        self.solver = solver
        self.device = next(model.parameters()).device
    
    def derivative(self, xt, t):
        # refer to Eq. (4) in EDM paper (https://arxiv.org/abs/2206.00364)
        st = self.scheduler.get_scaling(t)
        dst = self.scheduler.get_scaling_derivative(t)
        sigma_t = self.scheduler.get_sigma(t)
        dsigma_t = self.scheduler.get_sigma_derivative(t)
        return dst / st * xt - st * dsigma_t * sigma_t * self.model.score(xt/st, sigma=sigma_t)

    def sample(self, xT, num_steps=None, return_traj=False, requires_grad=False):
        # reverse PF-ODE, from prior Gaussian to data
        if num_steps is None:
            num_steps = self.scheduler.num_steps
        
        shape = xT.shape
        def _derivative_wrapper(t, xt):
            xt = xt.view(*shape)
            deriv = self.derivative(xt, t)
            return deriv.flatten(1)
        
        time_steps = self.scheduler.get_discrete_time_steps(num_steps).to(xT.device)
        if requires_grad:
            xT.requires_grad_(True)
            x_ode_traj = odeint_adjoint(_derivative_wrapper, xT.flatten(1), time_steps, rtol=1e-3, atol=1e-3, method=self.solver, adjoint_params=(xT))
        else:
            x_ode_traj = odeint(_derivative_wrapper, xT.flatten(1), time_steps, rtol=1e-3, atol=1e-3, method=self.solver) # [num_steps, B, D]
        x_ode_traj = x_ode_traj.view(num_steps, *shape)
        
        if return_traj:
            return x_ode_traj
        else:
            return x_ode_traj[-1]
        
    def inverse(self, x0, num_steps=None, return_traj=False):
        # forward PF-ODE, from data to prior Gaussian
        if num_steps is None:
            num_steps = self.scheduler.num_steps
        
        shape = x0.shape
        def _derivative_wrapper(t, xt):
            xt = xt.view(*shape)
            deriv = self.derivative(xt, t)
            return deriv.flatten(1)
        
        reverse_time_steps = self.scheduler.get_discrete_time_steps(num_steps).to(x0.device)
        # reverse timestep
        time_steps = reverse_time_steps.flip(0)
        x_ode_traj = odeint(_derivative_wrapper, x0.flatten(1), time_steps, method=self.solver)
        x_ode_traj = x_ode_traj.view(num_steps, *shape)

        if return_traj:
            return x_ode_traj
        else:
            return x_ode_traj[-1]

    def hutchinson_trace_estimate(self, x, t, num_random_vector):
        trace_estimate = torch.zeros(x.shape[0]).to(x.device)

        for _ in range(0, num_random_vector):
            z = torch.randn_like(x)
            xt = x.clone().detach().requires_grad_(True)
            loss = (self.derivative(xt, t) * z).sum()
            trace_sample = (z * grad(loss, xt)[0]).flatten(1).sum(1)
            trace_estimate += trace_sample
        return trace_estimate / num_random_vector

    def log_likelihood(self, x0, num_steps=None, num_random_vector=10, verbose=False):
        # get ODE trajectory
        if num_steps is None:
            num_steps = self.scheduler.num_steps
        else:
            num_steps = num_steps + 1 # include the initial step
        traj = self.inverse(x0, num_steps, True)
        reverse_time_steps = self.scheduler.get_discrete_time_steps(num_steps).to(x0.device)
        time_steps = reverse_time_steps.flip(0)
        delta_times = time_steps[1:] - time_steps[:-1]
        delta_times = torch.cat([delta_times[:1], delta_times])

        # calculate log likelihood
        self.model.requires_grad_(True)
        trace = torch.zeros(x0.shape[0]).to(x0.device)
        pbar = tqdm.trange(num_steps) if verbose else range(num_steps)
        for idx in pbar:
            xt, t, dt = traj[idx], time_steps[idx], delta_times[idx]
            trace += self.hutchinson_trace_estimate(xt, t, num_random_vector) * dt
        self.model.requires_grad_(False)

        noise = traj[-1]
        normal_dist = torch.distributions.Normal(0, self.scheduler.get_prior_sigma())
        log_prob = normal_dist.log_prob(noise).flatten(1).sum(1)
        return log_prob - trace

    def bit_dim(self, x0, num_steps=None, num_random_vector=10, verbose=False):
        logp = self.log_likelihood(x0, num_steps, num_random_vector, verbose)
        bit_dim = - logp / np.log(2) / np.prod(x0.shape[1:]) + 7
        return bit_dim

    def get_start(self, batch_size):
        in_shape = self.model.get_in_shape()
        x_start = torch.randn(batch_size, *in_shape, device=self.device) * self.scheduler.get_prior_sigma()
        return x_start


class DiffusionSDE:
    """
    Diffusion Stochastic Differential Equation (Diffusion SDE) for sampling via forward and reverse SDE processes.
    """
    def __init__(self, model, scheduler, solver='euler'):
        self.model = model
        self.scheduler = scheduler
        self.solver = solver
        self.device = next(model.parameters()).device
        if solver != 'euler':
            raise NotImplementedError
    
    def forward_sde(self, x0, t, num_steps=None, return_traj=False):
        pass

    def reverse_sde(self, x0, num_steps=None, return_traj=False):
        pass

    def get_start(self, batch_size):
        in_shape = self.model.get_in_shape()
        x_start = torch.randn(batch_size, *in_shape, device=self.device) * self.scheduler.get_prior_sigma()
        return x_start
