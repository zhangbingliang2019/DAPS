from piq import psnr, ssim, LPIPS
import prettytable
import torch
import torch.nn as nn
import wandb
import numpy as np
from abc import ABC, abstractmethod


class Metrics(nn.Module):
    def __init__(self, x0, op, y, eval_fn=('meas_error', 'psnr', 'ssim', 'lpips')):
        super().__init__()
        '''x0, y:[B, C, H, W]'''
        self.x0 = x0[None]
        self.op = op
        self.y = y[None]
        self.eval_fn = {}
        self.cmp_fn = {}
        if 'meas_error' in eval_fn:
            self.eval_fn['meas_error'] = lambda x, y: self.op.error(x, y)**(1/2)
            self.cmp_fn['meas_error'] = 'min'
        if 'psnr' in eval_fn:
            self.eval_fn['psnr'] = lambda x1, x2: psnr(x1, x2, data_range=1.0, reduction='none')
            self.cmp_fn['psnr'] = 'max'
        if 'ssim' in eval_fn:
            self.eval_fn['ssim'] = lambda x1, x2: ssim(x1, x2, data_range=1.0, reduction='none')
            self.cmp_fn['ssim'] = 'max'
        if 'lpips' in eval_fn:
            self.eval_fn['lpips'] = LPIPS(replace_pooling=True, reduction='none')
            self.cmp_fn['lpips'] = 'min'
        if 'lpips_max' in eval_fn:
            self.eval_fn['lpips_max'] = LPIPS(reduction='none')
            self.cmp_fn['lpips_max'] = 'min'
        if 'l2' in eval_fn:
            self.eval_fn['l2'] = lambda x1, x2: ((x1 - x2) ** 2).flatten(1).mean(-1)
            self.cmp_fn['l2'] = 'min'

    def norm(self, x):
        return (x * 0.5 + 0.5).clip(0, 1)

    def to_list(self, x):
        return x.cpu().detach().tolist()

    def eval(self, x):
        '''x: [N, B, C, H, W] or [B, C, H, W]'''
        if len(x.shape) == 4:
            x = x[None]
        result_dicts = {}

        # # measurement error
        # meas_error = ((self.op(x) - y) ** 2).flatten(2).sum(-1)
        # meas_error_mean = meas_error.mean(0)
        # meas_error_std = meas_error.std(0)
        # result_dicts['meas_error'] = {'mean': self.to_list(meas_error), 'std': self.to_list(meas_error_std)}

        # eval function
        broadcasted_shape = torch.broadcast_shapes(x.shape, self.x0.shape)
        x_flatten = x.expand(broadcasted_shape).flatten(0, 1)
        x0_flatten = self.x0.expand(broadcasted_shape).flatten(0, 1)
        y_flatten = self.y.expand((broadcasted_shape[0], *self.y.shape[1:])).flatten(0, 1)
        # print(broadcasted_shape)
        # print(x_flatten.shape, x0_flatten.shape)

        for key, fn in self.eval_fn.items():
            if key == 'meas_error':
                x_flatten_cur = x_flatten
                target_cur = y_flatten
            else:
                x_flatten_cur = self.norm(x_flatten)
                target_cur = self.norm(x0_flatten)
            value = fn(x_flatten_cur, target_cur).reshape(broadcasted_shape[0], -1)
            result_dicts[key] = {
                'sample': self.to_list(value.permute(1, 0)),
                'mean': self.to_list(value.mean(0)),
                'std': self.to_list(value.std(0) if value.shape[0] != 1 else torch.zeros_like(value.mean(0))),
                'max': self.to_list(value.max(0)[0]),
                'min': self.to_list(value.min(0)[0]),
            }

        # diversity
        # result_dicts['diversity'] = {'mean': self.to_list(x.mean(0)), 'std': self.to_list(x.std(0))}
        # outlier = (x.std(0) / (self.x0[0] - x.mean(0)).abs() > 3)
        # result_dicts['outlier'] = {'map': self.to_list(outlier), 'ratio': outlier.float().mean().item()}

        return result_dicts

    def display_old(self, result_dicts, std=False):
        table = Table('results')
        for key in result_dicts.keys():
            if not std:
                value = ['{:.2f}'.format(v) for v in result_dicts[key]['mean']]
            else:
                value = ['{:.2f} ({:.2f})'.format(v, f) for v, f in
                         zip(result_dicts[key]['mean'], result_dicts[key]['std'])]
            table.add_column(key, value)
        # print(table.table)
        return table.get_string()

    def display(self, result_dicts):
        table = Table('results')
        for key in result_dicts.keys():
            value = ['{:.2f}'.format(v) for v in result_dicts[key][self.cmp_fn[key]]]
            # print('KEY', key)
            # print('VALUE', len(value))
            table.add_column(key, value)
        # print(table.table)
        return table.get_string()

    def log_wandb(self, result_dicts, batch_size):

        for s in range(batch_size):
            log_dict = {key: result_dicts[key][self.cmp_fn[key]][s] for key in result_dicts.keys()}
            wandb.log(log_dict)
        log_dict = {key: np.mean(result_dicts[key][self.cmp_fn[key]]) for key in result_dicts.keys()}
        new_log_dict = {key + '_all': value for key, value in log_dict.items()}
        wandb.log(new_log_dict)
        return


class Table(object):
    def __init__(self, title=None, field_names=None):
        """
            title:          str
            field_names:    list of field names
        """
        self.table = prettytable.PrettyTable(title=title, field_names=field_names)

    def add_rows(self, rows):
        """
            rows: list of tuples
        """
        self.table.add_rows(rows)

    def add_row(self, row):
        self.table.add_row(row)

    def add_column(self, fieldname, column):
        self.table.add_column(fieldname=fieldname, column=column)

    def get_string(self):
        """
            a markdown format table
        """
        _junc = self.table.junction_char
        if _junc != "|":
            self.table.junction_char = "|"
        markdown = [row for row in self.table.get_string().split("\n")[1:-1]]
        self.table.junction_char = _junc
        return "\n" + "\n".join(markdown)

    def get_latex_string(self):
        # TODO: to be done in future
        pass



__EVAL_FN__ = {}


def register_eval_fn(name: str):
    def wrapper(cls):
        if __EVAL_FN__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __EVAL_FN__[name] = cls
        return cls

    return wrapper


def get_eval_fn(name: str, **kwargs):
    if __EVAL_FN__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __EVAL_FN__[name](**kwargs)


class EvalFn(torch.nn.Module):
    def __init__(self, gt, measurement):
        super().__init__()
        self.gt = gt
        self.measurement = measurement

    def norm(self, x):
        return (x * 0.5 + 0.5).clip(0, 1)

    def forward(self, sample):
        return self.evaluate(self.gt, self.measurement, sample)

    def evaluate(self, gt, measurement, sample):
        pass

@register_eval_fn('psnr')
class PeakSignalNoiseRatio(EvalFn):
    def evaluate(self, gt, measurement, sample):
        return psnr(self.norm(gt), self.norm(sample), 1.0, reduction='none')


@register_eval_fn('ssim')
class StructuralSimilarityIndexMeasure(EvalFn):
    def evaluate(self, gt, measurement, sample):
        return ssim(self.norm(gt), self.norm(sample), 1.0, reduction='none')

@register_eval_fn('lpips')
class LearnedPerceptualImagePatchSimilarity(EvalFn):
    def __init__(self, gt, measurement):
        super().__init__(gt, measurement)
        self.lpips_fn = LPIPS(replace_pooling=True, reduction='none')
    def evaluate(self, gt, measurement, sample):
        return self.lpips_fn(self.norm(gt), self.norm(sample))

