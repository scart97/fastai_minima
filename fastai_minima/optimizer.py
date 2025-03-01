# flake8: noqa

__all__ = [
    "detuplify_pg",
    "set_item_pg",
    "pytorch_hp_map",
    "OptimWrapper",
    "Adam",
    "SGD",
    "params",
    "convert_params",
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

# Cell
from collections import defaultdict

import torch
from fastcore.basics import GetAttr, even_mults, merge, range_of
from fastcore.foundation import L
from fastcore.meta import delegates
from fastcore.xtras import is_listy
from torch import optim

from .utils import tensor


# Cell
class _BaseOptimizer:
    "Common functionality between `Optimizer` and `OptimWrapper`"

    def all_params(self, n=slice(None), with_grad=False):
        "List of param_groups, paramters, and hypers"
        res = L(
            (p, pg, self.state[p], hyper)
            for pg, hyper in zip(self.param_lists[n], self.hypers[n])
            for p in pg
        )
        return (
            L(o for o in res if hasattr(o[0], "grad") and o[0].grad is not None)
            if with_grad
            else res
        )

    def _set_require_grad(self, rg, p, pg, state, h):
        p.requires_grad_(rg or state.get("force_train", False))

    def freeze_to(self, n):
        "Freeze parameter groups up to `n`"
        self.frozen_idx = n if n >= 0 else len(self.param_lists) + n
        if self.frozen_idx >= len(self.param_lists):
            warn(
                f"Freezing {self.frozen_idx} groups; model has {len(self.param_lists)}; whole model is frozen."
            )
        for o in self.all_params(slice(n, None)):
            self._set_require_grad(True, *o)
        for o in self.all_params(slice(None, n)):
            self._set_require_grad(False, *o)

    def freeze(self):
        "Freeze up to last parameter group"
        assert len(self.param_lists) > 1
        self.freeze_to(-1)

    def set_freeze(self, n, rg, ignore_force_train=False):
        "Set `rg` for parameter group `n` only"
        for p in self.param_lists[n]:
            p.requires_grad_(
                rg or (state.get("force_train", False) and not ignore_force_train)
            )

    def unfreeze(self):
        "Unfreeze the entire model"
        self.freeze_to(0)

    def set_hypers(self, **kwargs):
        "Apply `set_hyper` for all `kwargs`"
        L(kwargs.items()).starmap(self.set_hyper)

    def _set_hyper(self, k, v):
        for v_, h in zip(v, self.hypers):
            h[k] = v_

    def set_hyper(self, k, v):
        "Set the value(s) in `v` for hyper-paramter `k`"
        if isinstance(v, slice):
            if v.start:
                v = even_mults(v.start, v.stop, len(self.param_lists))
            else:
                v = [v.stop / 10] * (len(self.param_lists) - 1) + [v.stop]
        v = L(v, use_list=None)
        if len(v) == 1:
            v = v * len(self.param_lists)
        assert len(v) == len(
            self.hypers
        ), f"Trying to set {len(v)} values for {k} but there are {len(self.param_lists)} parameter groups."
        self._set_hyper(k, v)

    @property
    def param_groups(self):
        return [
            {**{"params": pg}, **hp} for pg, hp in zip(self.param_lists, self.hypers)
        ]

    @param_groups.setter
    def param_groups(self, v):
        for pg, v_ in zip(self.param_lists, v):
            pg = v_["params"]
        for hyper, v_ in zip(self.hypers, v):
            for k, t in v_.items():
                if k != "params":
                    hyper[k] = t


# Cell
def _update(state, new=None):
    if new is None:
        return state
    if isinstance(new, dict):
        state.update(new)
    return state


# Cell
def detuplify_pg(d):
    res = {}
    for k, v in d.items():
        if k == "params":
            continue
        if is_listy(v):
            res.update(**{f"{k}__{i}": v_ for i, v_ in enumerate(v)})
        else:
            res[k] = v
    return res


# Cell
def set_item_pg(pg, k, v):
    if "__" not in k:
        pg[k] = v
    else:
        name, idx = k.split("__")
        pg[name] = tuple(
            v if i == int(idx) else pg[name][i] for i in range_of(pg[name])
        )
    return pg


# Cell
pytorch_hp_map = {
    "momentum": "mom",
    "weight_decay": "wd",
    "alpha": "sqr_mom",
    "betas__0": "mom",
    "betas__1": "sqr_mom",
}

# Cell
class OptimWrapper(_BaseOptimizer, GetAttr):
    _xtra = ["zero_grad", "step", "state_dict", "load_state_dict"]
    _default = "opt"

    def __init__(self, opt, hp_map=None):
        self.opt = opt
        if hp_map is None:
            hp_map = pytorch_hp_map
        self.fwd_map = {
            k: hp_map[k] if k in hp_map else k
            for k in detuplify_pg(opt.param_groups[0]).keys()
        }
        self.bwd_map = {v: k for k, v in self.fwd_map.items()}
        self.state = defaultdict(dict, {})
        self.frozen_idx = 0

    @property
    def hypers(self):
        return [
            {self.fwd_map[k]: v for k, v in detuplify_pg(pg).items() if k != "params"}
            for pg in self.opt.param_groups
        ]

    def _set_hyper(self, k, v):
        for pg, v_ in zip(self.opt.param_groups, v):
            pg = set_item_pg(pg, self.bwd_map[k], v_)

    def clear_state(self):
        self.opt.state = defaultdict(dict, {})

    @property
    def param_lists(self):
        return [pg["params"] for pg in self.opt.param_groups]

    @param_lists.setter
    def param_lists(self, v):
        for pg, v_ in zip(self.opt.param_groups, v):
            pg["params"] = v_


# Cell
@delegates(optim.Adam)
def Adam(params, **kwargs):
    "Convience function to make an Adam optimizer compatable with `Learner`"
    return OptimWrapper(optim.Adam(params, **kwargs))


# Cell
@delegates(optim.SGD)
def SGD(params, **kwargs):
    "Convience function to make a SGD optimizer compatable with `Learner`"
    return OptimWrapper(optim.SGD(params, **kwargs))


# Cell
def params(m):
    "Return all parameters of `m`"
    return [p for p in m.parameters()]


# Cell
from torch import nn


def convert_params(o: list) -> list:
    """
    Converts `o` into Pytorch-compatable param groups

    `o` should be a set of layer-groups that should be split in the optimizer

    Example:

    ```python
    def splitter(m): return convert_params([[m.a], [m.b]])
    ```

    Where `m` is a model defined as:

    ```python
    class RegModel(Module):
      def __init__(self): self.a,self.b = nn.Parameter(torch.randn(1)),nn.Parameter(torch.randn(1))
      def forward(self, x): return x*self.a + self.b
    ```
    """
    if not isinstance(o[0], dict):
        splitter = []
        for group in o:
            if not isinstance(group[0], nn.parameter.Parameter):
                group = L(group).map(params)
            splitter.append({"params": group})
        return splitter
    return o
