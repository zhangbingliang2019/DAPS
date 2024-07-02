from piq import psnr, ssim, LPIPS
import prettytable
import torch
import torch.nn as nn
import wandb
import numpy as np


class Evaluator(nn.Module):
    """
        Evaluation module for computing evaluation metrics.
    """

    def __init__(self, gt, measurement, eval_fn_lists=('psnr', 'lpips')):
        """
            Initializes the evaluator with the ground truth and measurement.

            Parameters:
                gt (torch.Tensor): Ground truth tensor.
                measurement (torch.Tensor): Measurement tensor.
                eval_fn_lists (tuple): List of evaluation functions to use.
        """
        super().__init__()
        self.eval_fn = torch.nn.ModuleDict()
        for eval_fn_name in eval_fn_lists:
            self.eval_fn.add_module(eval_fn_name, get_eval_fn(eval_fn_name, gt=gt, measurement=measurement))
        self.main_eval_fn_name = eval_fn_lists[0]

    def forward(self, x, reduction='mean'):
        """
            Computes evaluation metrics for the given input.

            Parameters:
                x (torch.Tensor): Input tensor.
                reduction (str): Reduction method ('mean' or 'none').

            Returns:
                dict: Dictionary of evaluation results.
        """
        results = {}
        for eval_fn_name, eval_fn in self.eval_fn.items():
            if reduction == 'mean':
                results[eval_fn_name] = eval_fn(x).mean()
            else:
                results[eval_fn_name] = eval_fn(x)
        return results

    def set_batch_index(self, start, end):
        for eval_fn in self.eval_fn.values():
            eval_fn.set_batch_index(start, end)

    def to_list(self, x):
        return x.cpu().detach().tolist()

    def evaluate(self, gt, measurement, x):
        '''x: [N, B, C, H, W] or [B, C, H, W]'''
        if len(x.shape) == 4:
            x = x[None]
        result_dicts = {}

        # eval function
        broadcasted_shape = torch.broadcast_shapes(x.shape, gt.shape)
        x0_flatten = gt.expand(broadcasted_shape).flatten(0, 1)
        x_flatten = x.expand(broadcasted_shape).flatten(0, 1)
        y_flatten = measurement.expand((broadcasted_shape[0], *measurement.shape)).flatten(0, 1)

        for key, fn in self.eval_fn.items():
            value = fn.evaluate(x0_flatten, y_flatten, x_flatten).reshape(broadcasted_shape[0], -1)
            result_dicts[key] = {
                'sample': self.to_list(value.permute(1, 0)),
                'mean': self.to_list(value.mean(0)),
                'std': self.to_list(value.std(0) if value.shape[0] != 1 else torch.zeros_like(value.mean(0))),
                'max': self.to_list(value.max(0)[0]),
                'min': self.to_list(value.min(0)[0]),
            }
        return result_dicts

    def display(self, result_dicts):
        table = Table('results')
        summary = {}
        for key in result_dicts.keys():
            value = ['{:.3f}'.format(v) for v in result_dicts[key][get_eval_fn_cmp(key)]]
            table.add_column(key, value)
            summary[key] = '{:.3f}'.format(np.mean(result_dicts[key][get_eval_fn_cmp(key)]))
        # for average
        table.add_row(['' for _ in result_dicts.keys()])
        table.add_row(['avg' for _ in result_dicts.keys()])
        table.add_row(summary.values())

        return table.get_string()

    def log_wandb(self, result_dicts, batch_size):
        for s in range(batch_size):
            log_dict = {key: result_dicts[key][get_eval_fn_cmp(key)][s] for key in result_dicts.keys()}
            wandb.log(log_dict)
        log_dict = {key: np.mean(result_dicts[key][get_eval_fn_cmp(key)]) for key in result_dicts.keys()}
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
__EVAL_FN_CMP__ = {}

def register_eval_fn(name: str):
    def wrapper(cls):
        if __EVAL_FN__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __EVAL_FN__[name] = cls
        __EVAL_FN_CMP__[name] = cls.cmp
        return cls

    return wrapper


def get_eval_fn(name: str, **kwargs):
    if __EVAL_FN__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __EVAL_FN__[name](**kwargs)


def get_eval_fn_cmp(name: str):
    return __EVAL_FN_CMP__[name]


class EvalFn(torch.nn.Module):
    def __init__(self, gt, measurement):
        super().__init__()
        self.gt = gt
        self.measurement = measurement
        self.start = None
        self.end = None

    def norm(self, x):
        return (x * 0.5 + 0.5).clip(0, 1)

    def set_batch_index(self, start, end):
        """
            used before calling forward
        """
        self.start = start
        self.end = end

    def forward(self, sample):
        """
            calling before setting correct batch index
        """
        return self.evaluate(self.gt[self.start:self.end], self.measurement[self.start:self.end], sample)

    def evaluate(self, gt, measurement, sample):
        pass

@register_eval_fn('psnr')
class PeakSignalNoiseRatio(EvalFn):
    cmp = 'max'  # the higher, the better
    def evaluate(self, gt, measurement, sample):
        return psnr(self.norm(gt), self.norm(sample), 1.0, reduction='none')


@register_eval_fn('ssim')
class StructuralSimilarityIndexMeasure(EvalFn):
    cmp = 'max'  # the higher, the better
    def evaluate(self, gt, measurement, sample):
        return ssim(self.norm(gt), self.norm(sample), 1.0, reduction='none')

@register_eval_fn('lpips')
class LearnedPerceptualImagePatchSimilarity(EvalFn):
    cmp = 'min'  # the higher, the better
    def __init__(self, gt, measurement):
        super().__init__(gt, measurement)
        self.lpips_fn = LPIPS(replace_pooling=True, reduction='none')
    def evaluate(self, gt, measurement, sample):
        return self.lpips_fn(self.norm(gt), self.norm(sample))

