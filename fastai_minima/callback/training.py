# flake8: noqa

__all__ = [
    "annealer",
    "sched_lin",
    "sched_cos",
    "sched_no",
    "sched_exp",
    "SchedLin",
    "SchedCos",
    "SchedNo",
    "SchedExp",
    "SchedPoly",
    "combine_scheds",
    "combined_cos",
    "ParamScheduler",
    "LRFinder",
    "SuggestedLRs",
]

# Cell
# Contains code used/modified by fastai_minima author from fastai
# Copyright 2019 the fast.ai team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language

import collections
import functools
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# Cell
from fastcore.basics import patch, store_attr
from fastcore.foundation import L
from fastcore.meta import delegates
from fastcore.xtras import is_listy

from ..learner import Learner, Recorder
from ..optimizer import convert_params
from ..utils import defaults, params, tensor
from .core import Callback, CancelFitException, CancelValidException


# Cell
class _Annealer:
    def __init__(self, f, start, end):
        store_attr("f,start,end")

    def __call__(self, pos):
        return self.f(self.start, self.end, pos)


# Cell
def annealer(f):
    "Decorator to make `f` return itself partially applied."

    @functools.wraps(f)
    def _inner(start, end):
        return _Annealer(f, start, end)

    return _inner


# Cell
# TODO Jeremy, make this pickle
# @annealer
# def SchedLin(start, end, pos): return start + pos*(end-start)
# @annealer
# def SchedCos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
# @annealer
# def SchedNo (start, end, pos): return start
# @annealer
# def SchedExp(start, end, pos): return start * (end/start) ** pos
#
# SchedLin.__doc__ = "Linear schedule function from `start` to `end`"
# SchedCos.__doc__ = "Cosine schedule function from `start` to `end`"
# SchedNo .__doc__ = "Constant schedule function with `start` value"
# SchedExp.__doc__ = "Exponential schedule function from `start` to `end`"

# Cell
def sched_lin(start, end, pos):
    return start + pos * (end - start)


def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


def sched_no(start, end, pos):
    return start


def sched_exp(start, end, pos):
    return start * (end / start) ** pos


def SchedLin(start, end):
    "Linear schedule function from `start` to `end`"
    return _Annealer(sched_lin, start, end)


def SchedCos(start, end):
    "Cosine schedule function from `start` to `end`"
    return _Annealer(sched_cos, start, end)


def SchedNo(start, end):
    "Constant schedule function with `start` value"

    return _Annealer(sched_no, start, end)


def SchedExp(start, end):
    "Exponential schedule function from `start` to `end`"
    return _Annealer(sched_exp, start, end)


# Cell
def SchedPoly(start, end, power):
    "Polynomial schedule (of `power`) function from `start` to `end`"

    def _inner(pos):
        return start + (end - start) * pos ** power

    return _inner


# Cell
def combine_scheds(pcts, scheds):
    "Combine `scheds` according to `pcts` in one function"
    assert sum(pcts) == 1.0
    pcts = tensor([0] + L(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        if int(pos) == 1:
            return scheds[-1](1.0)
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos.item())

    return _inner


# Cell
def combined_cos(pct, start, middle, end):
    "Return a scheduler with cosine annealing from `start`→`middle` & `middle`→`end`"
    return combine_scheds(
        [pct, 1 - pct], [SchedCos(start, middle), SchedCos(middle, end)]
    )


# Cell
class ParamScheduler(Callback):
    "Schedule hyper-parameters according to `scheds`"
    order, run_valid = 60, False

    def __init__(self, scheds):
        self.scheds = scheds

    def before_fit(self):
        "Initialize container for hyper-parameters"
        self.hps = {p: [] for p in self.scheds.keys()}

    def before_batch(self):
        "Set the proper hyper-parameters in the optimizer"
        self._update_val(self.pct_train)

    def _update_val(self, pct):
        for n, f in self.scheds.items():
            self.opt.set_hyper(n, f(pct))

    def after_batch(self):
        "Record hyper-parameters of this batch"
        for p in self.scheds.keys():
            self.hps[p].append(self.opt.hypers[-1][p])

    def after_fit(self):
        "Save the hyper-parameters in the recorder if there is one"
        if hasattr(self.learn, "recorder") and hasattr(self, "hps"):
            self.recorder.hps = self.hps


# Cell
@patch
def fit_one_cycle(
    self: Learner,
    n_epoch,
    lr_max=None,
    div=25.0,
    div_final=1e5,
    pct_start=0.25,
    wd=None,
    moms=None,
    cbs=None,
    reset_opt=False,
):
    "Fit `self.model` for `n_epoch` using the 1cycle policy."
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper("lr", self.lr if lr_max is None else lr_max)
    lr_max = np.array([h["lr"] for h in self.opt.hypers])
    scheds = {
        "lr": combined_cos(pct_start, lr_max / div, lr_max, lr_max / div_final),
        "mom": combined_cos(pct_start, *(self.moms if moms is None else moms)),
    }
    self.fit(n_epoch, cbs=ParamScheduler(scheds) + L(cbs), reset_opt=reset_opt, wd=wd)


# Cell
@patch
def plot_sched(self: Recorder, keys=None, figsize=None):
    keys = self.hps.keys() if keys is None else L(keys)
    rows, cols = (len(keys) + 1) // 2, min(2, len(keys))
    figsize = figsize or (6 * cols, 4 * rows)
    _, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten() if len(keys) > 1 else L(axs)
    for p, ax in zip(keys, axs):
        ax.plot(self.hps[p])
        ax.set_ylabel(p)


# Cell
@patch
def fit_flat_cos(
    self: Learner,
    n_epoch,
    lr=None,
    div_final=1e5,
    pct_start=0.75,
    wd=None,
    cbs=None,
    reset_opt=False,
):
    "Fit `self.model` for `n_epoch` at flat `lr` before a cosine annealing."
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper("lr", self.lr if lr is None else lr)
    lr = np.array([h["lr"] for h in self.opt.hypers])
    scheds = {"lr": combined_cos(pct_start, lr, lr, lr / div_final)}
    self.fit(n_epoch, cbs=ParamScheduler(scheds) + L(cbs), reset_opt=reset_opt, wd=wd)


# Cell
@patch
def fit_sgdr(
    self: Learner,
    n_cycles,
    cycle_len,
    lr_max=None,
    cycle_mult=2,
    cbs=None,
    reset_opt=False,
    wd=None,
):
    "Fit `self.model` for `n_cycles` of `cycle_len` using SGDR."
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper("lr", self.lr if lr_max is None else lr_max)
    lr_max = np.array([h["lr"] for h in self.opt.hypers])
    n_epoch = cycle_len * (cycle_mult ** n_cycles - 1) // (cycle_mult - 1)
    pcts = [cycle_len * cycle_mult ** i / n_epoch for i in range(n_cycles)]
    scheds = [SchedCos(lr_max, 0) for _ in range(n_cycles)]
    scheds = {"lr": combine_scheds(pcts, scheds)}
    self.fit(n_epoch, cbs=ParamScheduler(scheds) + L(cbs), reset_opt=reset_opt, wd=wd)


# Cell
@patch
@delegates(Learner.fit_one_cycle)
def fine_tune(
    self: Learner,
    epochs,
    base_lr=2e-3,
    freeze_epochs=1,
    lr_mult=100,
    pct_start=0.3,
    div=5.0,
    **kwargs
):
    "Fine tune with `freeze` for `freeze_epochs` then with `unfreeze` from `epochs` using discriminative LR"
    self.freeze()
    self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
    base_lr /= 2
    self.unfreeze()
    self.fit_one_cycle(
        epochs,
        slice(base_lr / lr_mult, base_lr),
        pct_start=pct_start,
        div=div,
        **kwargs
    )


# Cell
class LRFinder(ParamScheduler):
    "Training with exponentially growing learning rate"

    def __init__(self, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
        if is_listy(start_lr):
            self.scheds = {"lr": [SchedExp(s, e) for (s, e) in zip(start_lr, end_lr)]}
        else:
            self.scheds = {"lr": SchedExp(start_lr, end_lr)}
        self.num_it, self.stop_div = num_it, stop_div

    def before_fit(self):
        "Initialize container for hyper-parameters and save the model"
        super().before_fit()
        self.learn.save("_tmp")
        self.best_loss = float("inf")

    def before_batch(self):
        "Record hyper-parameters of this batch and potentially stop training"
        self._update_val(self.train_iter / self.num_it)

    def after_batch(self):
        "Set the proper hyper-parameters in the optimizer"
        super().after_batch()
        if self.smooth_loss < self.best_loss:
            self.best_loss = self.smooth_loss
        if self.smooth_loss > 4 * self.best_loss and self.stop_div:
            raise CancelFitException()
        if self.train_iter >= self.num_it:
            raise CancelFitException()

    def before_validate(self):
        "Skip the validation part of training"
        raise CancelValidException()

    def after_fit(self):
        "Save the hyper-parameters in the recorder if there is one and load the original model"
        self.learn.opt.zero_grad()  # Need to zero the gradients of the model before detaching the optimizer for future fits
        tmp_f = self.path / self.model_dir / "_tmp.pth"
        if tmp_f.exists():
            self.learn.load("_tmp", with_opt=True)
            os.remove(tmp_f)


# Cell
@patch
def plot_lr_find(self: Recorder, skip_end=5):
    "Plot the result of an LR Finder test (won't work if you didn't do `learn.lr_find()` before)"
    lrs = self.lrs if skip_end == 0 else self.lrs[:-skip_end]
    losses = self.losses if skip_end == 0 else self.losses[:-skip_end]
    fig, ax = plt.subplots(1, 1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale("log")


# Cell
SuggestedLRs = collections.namedtuple("SuggestedLRs", ["lr_min", "lr_steep"])

# Cell
@patch
def lr_find(
    self: Learner,
    start_lr=1e-7,
    end_lr=10,
    num_it=100,
    stop_div=True,
    show_plot=True,
    suggestions=True,
):
    "Launch a mock training to find a good learning rate, return lr_min, lr_steep if `suggestions` is True"
    n_epoch = num_it // len(self.dls.train) + 1
    cb = LRFinder(start_lr=start_lr, end_lr=end_lr, num_it=num_it, stop_div=stop_div)
    with self.no_logging():
        self.fit(n_epoch, cbs=cb)
    if show_plot:
        self.recorder.plot_lr_find()
    if suggestions:
        lrs, losses = tensor(self.recorder.lrs[num_it // 10 : -5]), tensor(
            self.recorder.losses[num_it // 10 : -5]
        )
        if len(losses) == 0:
            return
        lr_min = lrs[losses.argmin()].item()
        grads = (losses[1:] - losses[:-1]) / (lrs[1:].log() - lrs[:-1].log())
        lr_steep = lrs[grads.argmin()].item()
        return SuggestedLRs(lr_min / 10.0, lr_steep)
