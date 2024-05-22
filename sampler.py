from abc import ABC, abstractmethod
import tqdm
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import torch
import numpy as np
import torch.nn as nn
from piq import psnr, LPIPS


def get_sampler(**kwargs):
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        return LatentDiffusionSampler(**kwargs)
    return DiffusionSampler(**kwargs)


class DiffusionSampler(nn.Module):
    def __init__(self, num_steps=200, solver='euler', scheduler='linear', timestep='poly-7', sigma_max=100,
                 sigma_min=0.01,
                 sigma_final=None, likelihood_estimator_config=None):
        super().__init__()
        self.num_steps = num_steps
        self.solver = solver
        self.scheduler = scheduler
        self.timestep = timestep
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_final = sigma_final
        if self.sigma_final is None:
            self.sigma_final = max(0, self.sigma_min - 0.01)

        if likelihood_estimator_config is None:
            self.likelihood_estimator = None
        else:
            self.likelihood_estimator = get_estimator(**likelihood_estimator_config)

        # get discretization
        steps = np.linspace(0, 1, num_steps)
        sigma_fn, sigma_derivative_fn, sigma_inv_fn = self.get_sigma_fn(self.scheduler)
        time_step_fn = self.get_time_step_fn(self.timestep, self.sigma_max, self.sigma_min)

        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, sigma_inv_fn(self.sigma_final))
        sigma_steps = np.array([sigma_fn(t) for t in time_steps])

        factor_steps = np.array(
            [2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for
             i in range(num_steps)])
        self.sigma_steps, self.time_steps, self.factor_steps = sigma_steps, time_steps, factor_steps

        # for safety issue
        self.factor_steps = [max(f, 2e-4) for f in self.factor_steps]
    def get_sigma_fn(self, scheduler):
        if scheduler == 'sqrt':
            sigma_fn = lambda t: np.sqrt(t)
            sigma_derivative_fn = lambda t: 1 / 2 / np.sqrt(t)
            sigma_inv_fn = lambda sigma: sigma ** 2

        elif scheduler == 'linear':
            sigma_fn = lambda t: t
            sigma_derivative_fn = lambda t: 1
            sigma_inv_fn = lambda t: t

        else:
            raise NotImplementedError
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_time_step_fn(self, timestep, sigma_max, sigma_min):
        if timestep == 'log':
            get_time_step_fn = lambda r: sigma_max ** 2 * (sigma_min ** 2 / sigma_max ** 2) ** r

        elif timestep.startswith('poly'):
            p = int(timestep.split('-')[1])
            get_time_step_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p

        # double_poly-{order}-{t_cut}-{first_sigma_min}-{second_sigma_max}
        elif timestep.startswith('double_poly'):
            # print('scheduler: ', timestep)
            tmp = timestep.split('-')[1:]
            p, t_cut, first_sigma_min, second_sigma_max = int(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3])
            def get_time_step_fn(r):
                if r < t_cut:
                    return (sigma_max ** (1 / p) + r / t_cut * (first_sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
                else:
                    return (second_sigma_max ** (1 / p) + (r - t_cut) / (1 - t_cut) * (sigma_min ** (1 / p) - second_sigma_max ** (1 / p))) ** p
        else:
            raise NotImplementedError
        return get_time_step_fn

    def _euler(self, model, x_start, op, y, SDE=False, verbose=False, record=False, checkpoint=False):
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        x = x_start
        for s in pbar:
            if checkpoint:
                prior_score = torch_checkpoint(model.score, x, self.sigma_steps[s])
            else:
                prior_score = model.score(x, self.sigma_steps[s])
            if record:
                self.sde_traj.add_image('prior_score', prior_score)
                self.sde_traj.add_value('sigma', self.sigma_steps[s])
            if self.likelihood_estimator is not None and op is not None and y is not None:
                likelihood_score, likelihood_weight = self.likelihood_estimator.noisy_likelihood_score(model, x, op, y,
                                                                                                       self.sigma_steps[
                                                                                                           s],
                                                                                                       s / self.num_steps)
                if record:
                    self.sde_traj.add_image('likelihood_score', likelihood_score)
                    self.sde_traj.add_value('likelihood_weight', likelihood_weight)
                    self.likelihood_estimator.record(self.sde_traj)
                score = prior_score + likelihood_weight * likelihood_score
            else:
                score = prior_score
                if record:
                    self.sde_traj.add_image('likelihood_score', torch.zeros_like(x))
                    self.sde_traj.add_value('likelihood_weight', 0)

            if SDE:
                epsilon = torch.randn_like(x)
                if record:
                    self.sde_traj.add_image('epsilon', epsilon)
                    self.sde_traj.add_image('xt', x)
                    self.sde_traj.add_value('factor', self.factor_steps[s])
                x = x + self.factor_steps[s] * score + np.sqrt(self.factor_steps[s]) * epsilon
            else:
                if record:
                    self.sde_traj.add_image('epsilon', torch.zeros_like(x))
                    self.sde_traj.add_image('xt', x)
                    self.sde_traj.add_value('factor', self.factor_steps[s])
                x = x + self.factor_steps[s] * score * 0.5
        if record:
            self.sde_traj.add_image('xt', x)
        return x

    def _resampling(self, model, x_start, op, y, SDE=False, verbose=False, record=False, checkpoint=False):
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        x = x_start
        for s in pbar:
            if record:
                self.sde_traj.add_image('xt', x)
            # x0hat = model.tweedie(x, self.sigma_steps[s])
            x0opt, _ = self.likelihood_estimator.noisy_likelihood_score(model, x, op, y, self.sigma_steps[s],
                                                                        s, self.num_steps, False)

            x = x0opt + self.sigma_steps[s + 1] * torch.randn_like(x0opt)

            if torch.isnan(x).any():
                return x

            if record:
                # self.sde_traj.add_image('tweedie', x0hat)
                self.sde_traj.add_image('x0opt', x0opt)
                self.likelihood_estimator.record(self.sde_traj)

        if record:
            self.sde_traj.add_image('xt', x)
        return x

    def sample(self, model, x_start, op=None, y=None, SDE=False, verbose=False, record=False, checkpoint=False):
        self.sde_traj = SDETrajectory()
        if self.solver == 'euler':
            return self._euler(model, x_start, op, y, SDE, verbose, record, checkpoint)
        elif self.solver == 'resampling':
            return self._resampling(model, x_start, op, y, SDE, verbose, record, checkpoint)
        # elif self.solver == 'heun':
        #     return self._heun(model, x_start, op, y, SDE, verbose)
        else:
            raise NotImplementedError

    def get_start(self, ref):
        x_start = torch.randn_like(ref) * self.sigma_max
        return x_start


class LatentDiffusionSampler(DiffusionSampler):
    def __init__(self, channel=3, down_factor=4, num_steps=200, solver='euler', scheduler='linear', timestep='poly-7',
                 sigma_max=150, sigma_min=0.01,
                 sigma_final=None, likelihood_estimator_config=None):
        super().__init__(num_steps, solver, scheduler, timestep, sigma_max, sigma_min, sigma_final,
                         likelihood_estimator_config)
        self.down_factor = down_factor
        self.channel = channel

    def _resampling(self, model, z_start, op, y, SDE=False, verbose=False, record=False, checkpoint=False):
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        z = z_start
        for s in pbar:
            # print('sigma: ', self.sigma_steps[s])
            if record:
                self.sde_traj.add_image('zt', z)
            # x0hat = model.tweedie(x, self.sigma_steps[s])
            z0opt, _ = self.likelihood_estimator.noisy_likelihood_score(model, z, op, y, self.sigma_steps[s], s,
                                                                        self.num_steps, True)
            z = z0opt + self.sigma_steps[s + 1] * torch.randn_like(z0opt)

            if torch.isnan(z).any():
                return z

            if record:
                self.sde_traj.add_image('z0opt', z0opt)
                self.likelihood_estimator.record(self.sde_traj)

        if record:
            self.sde_traj.add_image('zt', z)
        return z

    def sample(self, model, z_start, op=None, y=None, SDE=False, verbose=False, record=False, checkpoint=False,
               return_latent=False):
        self.sde_traj = SDETrajectory()
        if self.solver == 'euler':
            assert op is None and y is None
            z0 = self._euler(model, z_start, op, y, SDE, verbose, record, checkpoint)
        elif self.solver == 'resampling':
            z0 = self._resampling(model, z_start, op, y, SDE, verbose, record, checkpoint)
        # elif self.solver == 'heun':
        #     return self._heun(model, x_start, op, y, SDE, verbose)
        else:
            raise NotImplementedError

        if return_latent:
            return model.decode(z0), z0
        return model.decode(z0)

    def get_start(self, ref):
        shape = [ref.shape[0], self.channel, ref.shape[2] // self.down_factor, ref.shape[3] // self.down_factor]
        x_start = torch.randn(shape).to(ref.device) * self.sigma_max
        return x_start


class LikelihoodEstimator(ABC):
    @abstractmethod
    def noisy_likelihood_score(self, model, x, op, y, sigma, time_step, totoal_step, latent=False):
        # return likelihood_score, likelihood_weight
        pass

    def record(self, sde_traj):
        pass


__ESTIMATOR__ = {}


def register_estimator(name: str):
    def wrapper(cls):
        if __ESTIMATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __ESTIMATOR__[name] = cls
        return cls

    return wrapper


def get_estimator(name: str, **kwargs):
    if __ESTIMATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __ESTIMATOR__[name](**kwargs)


# Noisy likelihood estimator

@register_estimator(name='dps')
class DPS(LikelihoodEstimator):
    def __init__(self, weight=1.0):
        self.weight = weight

    def noisy_likelihood_score(self, model, x, op, y, sigma, time_ratio):
        x_t = x.clone().requires_grad_(True)
        x0hat = model.tweedie(x_t, sigma)
        log_likelihood = op.log_likelihood(x0hat, y)
        score = torch.autograd.grad(log_likelihood, x_t, grad_outputs=torch.ones_like(log_likelihood))[0]
        return score, self.weight


@register_estimator(name='ode_backward')
class ODEBackward(LikelihoodEstimator):
    def __init__(self, weight=1.0, checkpoint=False, ode_step=40):
        self.weight = weight
        self.x0 = None
        self.checkpoint = checkpoint
        self.ode_step = ode_step

    def noisy_likelihood_score(self, model, x, op, y, sigma, time_ratio):
        sampler = DiffusionSampler(num_steps=self.ode_step, solver='euler', scheduler='linear', timestep='poly-7',
                                   sigma_max=sigma, sigma_min=0.01)
        x_t = x.clone().requires_grad_(True)
        x0 = sampler.sample(model, x_t, SDE=False, verbose=False, checkpoint=self.checkpoint)
        self.x0 = x0
        log_likelihood = op.log_likelihood(x0, y)
        # score = torch.autograd.grad(log_likelihood, x_t, grad_outputs=torch.ones_like(log_likelihood), retain_graph=True)[0]
        log_likelihood.sum().backward()
        score = x_t.grad
        return score, self.weight

    def record(self, sde_traj):
        sde_traj.add_image('x0', self.x0)


@register_estimator(name='sde_backward')
class SDEBackward(LikelihoodEstimator):
    def __init__(self, N=3, weight=1.0, checkpoint=False, sde_step=40):
        self.weight = weight
        self.x0 = None
        self.checkpoint = checkpoint
        self.sde_step = sde_step
        self.N = N

    def noisy_likelihood_score(self, model, x, op, y, sigma, time_ratio):
        sampler = DiffusionSampler(num_steps=self.sde_step, solver='euler', scheduler='linear', timestep='poly-7',
                                   sigma_max=sigma, sigma_min=0.01)
        x_t = x.clone().requires_grad_(True)  # (B, 3, D, D)
        x_tn = x_t.unsqueeze(dim=1).repeat(1, self.N, 1, 1, 1).view(-1, *x_t.shape[1:])
        x0 = sampler.sample(model, x_tn, SDE=True, verbose=False, checkpoint=self.checkpoint)
        self.x0 = x0.view(-1, self.N, *x_t.shape[1:])
        y_n = y.unsqueeze(dim=1).repeat(1, self.N, 1, 1, 1).view(-1, *y.shape[1:])
        log_likelihood = op.log_likelihood(x0, y_n).view(-1, self.N).max(1)[0]
        # score = torch.autograd.grad(log_likelihood, x_t, grad_outputs=torch.ones_like(log_likelihood), retain_graph=True)[0]
        log_likelihood.sum().backward()
        score = x_t.grad
        return score, self.weight

    def record(self, sde_traj):
        sde_traj.add_image('x0', self.x0)


@register_estimator(name='optimization')
class Optimization(LikelihoodEstimator):
    def __init__(self, step=100, lr=0.1, weight=1.0, threshold='auto', ode_step=5):
        self.opt_steps = step
        self.lr = float(lr)
        self.threshold = threshold
        self.weight = weight
        self.x0hat = None
        self.ode_step = ode_step

    def noisy_likelihood_score(self, model, x, op, y, sigma, time_ratio):
        sampler = DiffusionSampler(num_steps=self.ode_step, solver='euler', scheduler='linear', timestep='poly-7',
                                   sigma_max=sigma, sigma_min=0.01, sigma_final=None)
        self.x0hat = sampler.sample(model, x, SDE=False)
        x0hat = self.x0hat.clone().detach().requires_grad_(True)
        # z = torch.zeros_like(x0hat).requires_grad_(True)
        optimizer = torch.optim.SGD([x0hat], lr=self.lr)

        for step in range(self.opt_steps):
            loss = op.error(x0hat, y).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x0hat.data = x0hat.data.clip(-1, 1)
                if self.threshold == 'auto':
                    threshold = sigma
                else:
                    threshold = self.threshold
                x0hat.data = x0hat.data.clip(self.x0hat - threshold, self.x0hat + threshold)
        return x0hat.detach(), self.weight

    def record(self, sde_traj):
        sde_traj.add_image('tweedie', self.x0hat)


#
# @register_estimator(name='langevin')
# class Langevin(LikelihoodEstimator):
#     def __init__(self, step=1000, lr='auto', ode_step=5, tau=0.05):
#         self.langevin_step = step
#         self.lr = lr
#         self.ode_step = ode_step
#         self.tau = tau
#         self.x0 = None
#
#     def noisy_likelihood_score(self, model, x, op, y, sigma, time_ratio):
#         if self.lr == 'auto':
#             lr = sigma ** 2 / 2
#         else:
#             lr = float(self.lr)
#         sampler = DiffusionSampler(num_steps=self.ode_step, solver='euler', scheduler='linear', timestep='poly-7',
#                                    sigma_max=sigma, sigma_min=0.02)
#
#         self.x0 = sampler.sample(model, x, SDE=False, verbose=False)
#         x = self.x0.clone().detach().requires_grad_(True)
#
#         optimizer = torch.optim.SGD([x], lr=lr)
#         for _ in range(self.langevin_step):
#             loss = op.error(x, y).sum() / self.tau ** 2 / 2
#             loss += ((x - self.x0) ** 2).sum() / sigma ** 2 / 2
#
#             optimizer.zero_grad()
#             loss.backward(retain_graph=True)
#             optimizer.step()
#             with torch.no_grad():
#                 x.data = x.data + np.sqrt(2 * lr) * torch.randn_like(x)
#                 x.data = x.data.clip(-1, 1)
#         return x.detach(), None
#
#     def record(self, sde_traj):
#         sde_traj.add_image('tweedie', self.x0)
#
#

@register_estimator(name='langevin')
class Langevin(LikelihoodEstimator):
    def __init__(self, step=100, lr=0.1, ode_step=10, tau=0.1, lr_scheduler='decay', space='pixel', rescale=1, milestone=1, return_tweedie=True, SDE=False, optimizer='sgd', lr_min_ratio=0.01):
        self.langevin_step = step
        self.lr = float(lr)
        self.ode_step = ode_step
        self.tau = float(tau)
        self.lr_scheduler = lr_scheduler
        self.space = space
        self.rescale = rescale
        self.milestone = milestone
        self.return_tweedie = return_tweedie
        self.optimizer = optimizer
        self.SDE = SDE
        self.lr_min_ratio = lr_min_ratio

        # self.eval
        self.psnr_fn = lambda x, x0: psnr(x, x0, 1.0, 'none')
        self.lpips_fn = LPIPS(True, reduction='none')

    def noisy_likelihood_score_test(self, model, x, op, y, sigma, time_step, totoal_step):
        sampler = DiffusionSampler(num_steps=self.ode_step, solver='euler', scheduler='linear', timestep='poly-7',
                                   sigma_max=sigma, sigma_min=0.01)
        with torch.no_grad():
            x0 = sampler.sample(model, x, SDE=False, verbose=False)
        x = x0.clone().detach().requires_grad_(True)
        self.x0 = x0
        lr = self.lr * sigma

        for _ in range(self.langevin_step):
            loss = op.error(x, y).sum() / self.tau ** 2 * 0
            loss += ((x - x0) ** 2).sum() / (2 * sigma ** 2)

            grad = torch.autograd.grad(loss, x)[0]
            # print(grad.max(), grad.min(), (grad ** 2).mean())
            # print(x.max(), x.min())
            # print()
            # with torch.no_grad():
            # x.data = x.data - lr * grad + np.sqrt(2 * lr) * torch.randn_like(x)
        return x.detach(), None

    def noisy_likelihood_score(self, model, x, op, y, sigma, time_step, totoal_step, latent=False):
        if latent:
            return self.latent_score(model, x, op, y, sigma, time_step, totoal_step)
        else:
            return self.pixel_score(model, x, op, y, sigma, time_step, totoal_step)

    def pixel_score(self, model, x, op, y, sigma, time_step, total_step):
        # print(sigma)
        sampler = DiffusionSampler(num_steps=self.ode_step, solver='euler', scheduler='linear', timestep='poly-7',
                                   sigma_max=sigma, sigma_min=0.01)
        with torch.no_grad():
            x0 = sampler.sample(model, x, SDE=self.SDE, verbose=False)
        self.x0 = x0
        if time_step != total_step - 1 or not self.return_tweedie:
            x = self.langevin_pixel(model, x0, sigma, op, y, time_step / total_step)
        else:
            x = x0
        return x, None

    def latent_score(self, model, z, op, y, sigma, time_step, total_step):
        sampler = LatentDiffusionSampler(num_steps=self.ode_step, solver='euler', scheduler='linear', timestep='poly-7',
                                         sigma_max=sigma, sigma_min=0.01)
        # if sigma < self.tau:
        #     return x, None
        with torch.no_grad():
            x0, z0 = sampler.sample(model, z, SDE=self.SDE, verbose=False, return_latent=True)

        self.x0 = x0
        self.z0 = z0
        #print(time_step, totoal_step)
        if time_step != total_step - 1 or not self.return_tweedie:
            if self.space == 'auto':
                # if time_step / totoal_step >= 2 / 3:
                if sigma <= self.milestone:
                    z = self.langevin_latent(model, z0, sigma, op, y, time_step/total_step)
                else:
                    x = self.langevin_pixel(model, x0, sigma, op, y, time_step/total_step)
                    z = model.encode(x)
            elif self.space == 'pixel':
                x = self.langevin_pixel(model, x0, sigma, op, y, time_step/total_step)
                self.x0opt = x
                z = model.encode(x)
            else:
                z = self.langevin_latent(model, z0, sigma, op, y, time_step/total_step)

        else:
            z = z0
        return z, None

    def get_optimizer(self, x, lr):
        if self.optimizer == 'sgd':
            return torch.optim.SGD([x], lr)
        elif self.optimizer == 'adam':
            return torch.optim.Adam([x], lr)

    def eval_pixel(self, x, x0, model, sigma, op, y):
        if not hasattr(self, 'gt'):
            return
        def norm(x):
            return (x * 0.5 + 0.5).clip(0, 1)
        with torch.no_grad():
            print()
            z0_psnr = self.psnr_fn(norm(self.gt), norm(x0))
            z0y_psnr = self.psnr_fn(norm(self.gt), norm(x))
            z0_lpips = self.lpips_fn(norm(self.gt), norm(x0))
            z0y_lpips = self.lpips_fn(norm(self.gt), norm(x))
            print('x0: ', x0.min(), x0.max(), x0.std())
            print('x: ', x.min(), x.max(), x.std())
            print('dist: ', (x0 - x).abs().min(), (x0 - x).abs().max(), (x0 - x).abs().std())
            print('x0 psnr: ', z0_psnr)
            print('x0y psnr: ', z0y_psnr)
            print('x0 lpips: ', z0_lpips)
            print('x0y lpips: ', z0y_lpips)
            print('psnr mean: ', z0y_psnr.mean())
            print('lpips mean: ', z0y_lpips.mean())
            print('meas error: ', op.error(x, y).sum())
            print('sigma: ', sigma)
            print('lr', self.lr)
            print()

    def eval_latent(self, z, z0, model, sigma, op, y):
        if not hasattr(self, 'gt'):
            return
        def norm(x):
            return (x * 0.5 + 0.5).clip(0, 1)
        with torch.no_grad():
            print()
            z0_psnr = self.psnr_fn(norm(self.gt), norm(model.decode(z0)))
            z0y_psnr = self.psnr_fn(norm(self.gt), norm(model.decode(z)))
            z0_lpips = self.lpips_fn(norm(self.gt), norm(model.decode(z0)))
            z0y_lpips = self.lpips_fn(norm(self.gt), norm(model.decode(z)))
            print('z0: ', z0.min(), z0.max(), z0.std())
            print('z: ', z.min(), z.max(), z.std())
            print('dist: ', (z0-z).abs().min(), (z0-z).abs().max(), (z0-z).abs().std())
            print('z0 psnr: ', z0_psnr)
            print('z0y psnr: ', z0y_psnr)
            print('z0 lpips: ', z0_lpips)
            print('z0y lpips: ', z0y_lpips)
            print('psnr mean: ', z0y_psnr.mean())
            print('lpips mean: ', z0y_lpips.mean())
            print('meas error: ', op.error(model.decode(z), y).sum())
            print('sigma: ', sigma)
            print('lr', self.lr)
            print()


    def get_learning_rate_multiplier(self, r):
        p = 1
        return (1 ** (1 / p) + r * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p

    def langevin_pixel(self, model, x0, sigma, op, y, ratio):
        x = x0.clone().detach().requires_grad_(True)
        if self.lr_scheduler == 'sigma':
            lr = self.lr * sigma
        elif self.lr_scheduler == 'none':
            lr = self.lr
        else:
            # print('Here')
            # lr = self.lr * sigma / self.sigma_max
            lr = self.lr * self.get_learning_rate_multiplier(ratio)
            # print('lr: ', lr)
        optimizer = self.get_optimizer(x, lr)

        for _ in range(self.langevin_step):
            optimizer.zero_grad()
            loss = op.error(x, y).sum() / self.tau ** 2
            loss += ((x - x0) ** 2).sum() / (2 * sigma ** 2)
            loss.backward()
            # print(x.grad.max() * lr, x.grad.min() * lr)
            # print(x.max(), x.min())
            # print()
            optimizer.step()
            with torch.no_grad():
                x.data = x.data + np.sqrt(2 * lr) * torch.randn_like(x)

            if torch.isnan(x).any():
                return x.detach()
        x.data = x.data.clip(-3, 3)
        self.eval_pixel(x, x0, model, sigma, op, y)
        return x.detach()

    def langevin_latent(self, model, z0, sigma, op, y, ratio):
        z = z0.clone().detach().requires_grad_(True)
        if self.lr_scheduler == 'sigma':
            lr = self.lr * sigma
        elif self.lr_scheduler == 'none':
            lr = self.lr
        else:
            # print('Here')
            lr = self.lr * self.get_learning_rate_multiplier(ratio)
        optimizer = self.get_optimizer(z, lr)

        for _ in range(self.langevin_step):
            optimizer.zero_grad()
            x = model.decode(z)
            loss = self.rescale * op.error(x, y).sum() / self.tau ** 2
            loss += ((z - z0) ** 2).sum() / (2 * sigma ** 2)
            loss.backward()
            # print(z.grad.max() * lr, z.grad.min() * lr)
            # print(z.max(), z.min())
            # print('latent')
            # print()
            optimizer.step()
            with torch.no_grad():
                z.data = z.data + np.sqrt(2 * lr) * torch.randn_like(z)

            if torch.isnan(z).any():
                return z.detach()
        z.data = z.data.clip(-10, 10)
        self.eval_latent(z, z0, model, sigma, op, y)
        return z.detach()

    def record(self, sde_traj):
        sde_traj.add_image('tweedie', self.x0)
        if hasattr(self, 'z0'):
            sde_traj.add_image('z0', self.z0)
        if hasattr(self, 'x0opt'):
            sde_traj.add_image('x0opt', self.x0opt)

@register_estimator(name='double_langevin')
class DoubleEstimator(LikelihoodEstimator):
    def __init__(self, t_cut, config1, config2):
        self.t_cut = t_cut
        self.lgv1 = get_estimator(**config1)
        self.lgv2 = get_estimator(**config2)
        self.ptr = 1

    def noisy_likelihood_score(self, model, x, op, y, sigma, time_step, totoal_step, latent=False):
        if time_step / totoal_step < self.t_cut:
            self.ptr = 1
            # print('pointer: ', self.ptr)
            return self.lgv1.noisy_likelihood_score(model, x, op, y, sigma, time_step, totoal_step, latent)
        else:
            self.ptr = 2
            # print('pointer: ', self.ptr)
            return self.lgv2.noisy_likelihood_score(model, x, op, y, sigma, time_step, totoal_step, latent)

    def record(self, sde_traj):
        if self.ptr == 1:
            # print('record: ', self.ptr)
            self.lgv1.record(sde_traj)
        else:
            # print('record: ', self.ptr)
            self.lgv2.record(sde_traj)



# @register_estimator(name='langevin_blackhole')
# class LangevinBlackHole(LikelihoodEstimator):
#     def __init__(self, step=100, lr=0.1, ode_step=10, tau=0.1, lr_scheduler='none', space='pixel'):
#         self.langevin_step = step
#         self.lr = float(lr)
#         self.ode_step = ode_step
#         self.tau = float(tau)
#         self.lr_scheduler = lr_scheduler
#         self.space = space
#
#     def noisy_likelihood_score(self, model, x, op, y, sigma, time_step, totoal_step, latent=False):
#         if latent:
#             raise NotImplementedError
#         else:
#             return self.pixel_score(model, x, op, y, sigma, time_step, totoal_step)
#
#     def pixel_score(self, model, x, op, y, sigma, time_step, totoal_step):
#         sampler = DiffusionSampler(num_steps=self.ode_step, solver='euler', scheduler='linear', timestep='poly-7',
#                                    sigma_max=sigma, sigma_min=0.01)
#         with torch.no_grad():
#             x0 = sampler.sample(model, x, SDE=False, verbose=False)
#         self.x0 = x0
#         if time_step != totoal_step - 1:
#             x = self.langevin_pixel(model, x0, sigma, op, y, time_step/totoal_step)
#         else:
#             x = x0
#         return x, None
#
#     def langevin_pixel(self, model, x0, sigma, op, y, ratio):
#         x = x0.clone().detach()
#         if self.lr_scheduler == 'sigma':
#             lr = self.lr * sigma
#         else:
#             # print('Here')
#             lr = self.lr * (1 - ratio)
#
#         for _ in range(self.langevin_step):
#             with torch.no_grad():
#                 grad = op.grad(x) / self.tau ** 2 + (x - x0) / sigma**2
#                 x -= lr * grad
#                 x += + np.sqrt(2 * lr) * torch.randn_like(x)
#
#             if torch.isnan(x).any():
#                 return x.detach()
#         return x.detach()
#
#     def record(self, sde_traj):
#         sde_traj.add_image('tweedie', self.x0)


@register_estimator(name='finetuning')
class Finetuning(LikelihoodEstimator):
    def __init__(self, opt_steps=10, opt_lr=1e-4, weight=1.0, threshold=0.05, ode_step=5):
        self.opt_steps = opt_steps
        self.lr = opt_lr
        self.threshold = threshold
        self.weight = weight
        self.x0hat = None
        self.ode_step = ode_step

    def noisy_likelihood_score(self, model, x, op, y, sigma, time_ratio):
        model.requires_grad_(True)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        sampler = DiffusionSampler(num_steps=self.ode_step, solver='euler', scheduler='linear', timestep='poly-7',
                                   sigma_max=sigma, sigma_min=0.01, sigma_final=None)

        for step in range(self.opt_steps + 1):
            self.x0hat = sampler.sample(model, x, SDE=False)
            x0hat = self.x0hat
            if step != self.opt_steps:
                loss = op.error(x0hat, y).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.x0hat.detach(), self.weight

    def record(self, sde_traj):
        sde_traj.add_image('tweedie', self.x0hat)


@register_estimator(name='sde_tree_backward')
class SDETreeBackward(LikelihoodEstimator):
    def __init__(self, N=3, weight=1.0, checkpoint=False, sde_step=5, rho=0.8):
        self.weight = weight
        self.x0 = None
        self.checkpoint = checkpoint
        self.sde_step = sde_step
        self.N = N
        self.rho = rho

    def noisy_likelihood_score(self, model, x, op, y, sigma, time_ratio):
        sampler = DiffusionSampler(num_steps=self.sde_step, solver='euler', scheduler='linear', timestep='poly-7',
                                   sigma_max=sigma, sigma_min=sigma * self.rho)
        x_t = x.clone().requires_grad_(True)  # (B, 3, D, D)
        x_tn = x_t.unsqueeze(dim=1).repeat(1, self.N, 1, 1, 1).view(-1, *x_t.shape[1:])
        x_t_prime = sampler.sample(model, x_tn, SDE=True, verbose=False, checkpoint=self.checkpoint)
        x0 = model.tweedie(x_t_prime, sigma * self.rho)
        self.x0 = x0.view(-1, self.N, *x_t.shape[1:])
        y_n = y.unsqueeze(dim=1).repeat(1, self.N, 1, 1, 1).view(-1, *y.shape[1:])
        log_likelihood = op.log_likelihood(x0, y_n).view(-1, self.N).max(1)[0]
        # score = torch.autograd.grad(log_likelihood, x_t, grad_outputs=torch.ones_like(log_likelihood), retain_graph=True)[0]
        log_likelihood.sum().backward()
        score = x_t.grad
        return score, self.weight

    def record(self, sde_traj):
        sde_traj.add_image('x0', self.x0)


class SDETrajectory(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_data = {}
        self.value_data = {}

    def add_image(self, name, images):
        if name not in self.image_data:
            self.image_data[name] = []
        self.image_data[name].append(images.detach().cpu())

    def add_value(self, name, values):
        if name not in self.value_data:
            self.value_data[name] = []
        self.value_data[name].append(values)

    def compile(self):
        for name in self.image_data.keys():
            self.image_data[name] = torch.stack(self.image_data[name], dim=0)
        for name in self.value_data.keys():
            self.value_data[name] = torch.tensor(self.value_data[name])

        # compute tweedie
        if not 'tweedie' in self.image_data.keys():
            prior_score = self.image_data['prior_score']
            xt = self.image_data['xt'][:-1]
            self.image_data['tweedie'] = xt + prior_score * self.value_data['sigma'].view(-1, 1, 1, 1, 1) ** 2
