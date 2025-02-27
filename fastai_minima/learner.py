# flake8: noqa

__all__ = [
    "DataLoaders",
    "replacing_yield",
    "mk_metric",
    "save_model",
    "load_model",
    "Learner",
    "before_batch_cb",
    "load_learner",
    "to_detach_from_dl",
    "Metric",
    "AvgMetric",
    "AvgLoss",
    "AvgSmoothLoss",
    "ValueMetric",
    "Recorder",
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
import pickle
import time
from builtins import NotImplementedError
from contextlib import contextmanager
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastcore.basics import GetAttr, Self, add_props, detuplify, patch, store_attr
from fastcore.foundation import L
from fastcore.meta import delegates
from fastcore.xtras import ContextManagers, Path, join_path_file

from .callback.core import (
    Callback,
    CancelBatchException,
    CancelEpochException,
    CancelFitException,
    CancelStepException,
    CancelTrainException,
    CancelValidException,
    GatherPredsCallback,
    TrainEvalCallback,
    event,
)
from .optimizer import Adam
from .utils import (
    NoneType,
    defaults,
    distrib_barrier,
    find_bs,
    find_device,
    get_model,
    noop,
    norm_bias_params,
    rank_distrib,
    tensor,
    to_detach,
    trainable_params,
)


# Cell
class DataLoaders(GetAttr):
    "Basic wrapper around several `DataLoader`s."
    _default = "train"

    def __init__(self, *loaders, path="."):
        self.loaders, self.path = list(loaders), Path(path)

    def __getitem__(self, i):
        return self.loaders[i]

    def new_empty(self):
        loaders = [dl.new(dl.dataset.new_empty()) for dl in self.loaders]
        return type(self)(*loaders, path=self.path, device=self.device)

    def _set(i, self, v):
        self.loaders[i] = v

    train, valid = add_props(lambda i, x: x[i], _set)
    train_ds, valid_ds = add_props(lambda i, x: x[i].dataset)

    def one_batch(self, ds_idx=0):
        "Grab one batch of data from the `DataLoader` at `ds_idx` in `self.loaders`"
        return next(iter(self.loaders[ds_idx]))


# Cell
defaults.lr = 1e-3

# Cell
def replacing_yield(o, attr, val):
    "Context manager to temporarily replace an attribute"
    old = getattr(o, attr)
    try:
        yield setattr(o, attr, val)
    finally:
        setattr(o, attr, old)


# Cell
def mk_metric(m):
    "Convert `m` to an `AvgMetric`, unless it's already a `Metric`"
    if isinstance(m, type):
        m = m()
    return m if isinstance(m, Metric) else AvgMetric(m)


# Cell
def save_model(file, model, opt, with_opt=True, pickle_protocol=2):
    "Save `model` to `file` along with `opt` (if available, and if `with_opt`)"
    if rank_distrib():
        return  # don't save if child proc
    if opt is None:
        with_opt = False
    state = get_model(model).state_dict()
    if with_opt:
        state = {"model": state, "opt": opt.state_dict()}
    torch.save(state, file, pickle_protocol=pickle_protocol)


# Cell
def load_model(file, model, opt, with_opt=True, device=None, strict=True):
    "Load `model` from `file` along with `opt` (if available, and if `with_opt`)"
    distrib_barrier()
    if isinstance(device, int):
        device = torch.device("cuda", device)
    elif device is None:
        device = "cpu"
    state = torch.load(file, map_location=device)
    hasopt = set(state) == {"model", "opt"}
    model_state = state["model"] if hasopt else state
    get_model(model).load_state_dict(model_state, strict=strict)
    if hasopt and with_opt:
        try:
            opt.load_state_dict(state["opt"])
        except:
            if with_opt:
                warn("Could not load the optimizer state.")
    elif with_opt:
        warn("Saved filed doesn't contain an optimizer state.")


# Cell
def _try_concat(o):
    try:
        return torch.cat(o)
    except:
        return sum([L(o_[i, :] for i in range_of(o_)) for o_ in o], L())


# Cell
_before_epoch = [event.before_fit, event.before_epoch]
_after_epoch = [event.after_epoch, event.after_fit]

# Cell
class _ConstantFunc:
    "Returns a function that returns `o`"

    def __init__(self, o):
        self.o = o

    def __call__(self, *args, **kwargs):
        return self.o


# Cell
_loop = [
    "Start Fit",
    "before_fit",
    "Start Epoch Loop",
    "before_epoch",
    "Start Train",
    "before_train",
    "Start Batch Loop",
    "before_batch",
    "after_pred",
    "after_loss",
    "before_backward",
    "before_step",
    "after_step",
    "after_cancel_batch",
    "after_batch",
    "End Batch Loop",
    "End Train",
    "after_cancel_train",
    "after_train",
    "Start Valid",
    "before_validate",
    "Start Batch Loop",
    "**CBs same as train batch**",
    "End Batch Loop",
    "End Valid",
    "after_cancel_validate",
    "after_validate",
    "End Epoch Loop",
    "after_cancel_epoch",
    "after_epoch",
    "End Fit",
    "after_cancel_fit",
    "after_fit",
]

# Cell
class Learner(GetAttr):
    _default = "model"

    def __init__(
        self,
        dls,
        model,
        loss_func=None,
        opt_func=Adam,
        lr=defaults.lr,
        splitter=trainable_params,
        cbs=None,
        metrics=None,
        path=None,
        model_dir="models",
        wd=None,
        wd_bn_bias=False,
        train_bn=True,
        moms=(0.95, 0.85, 0.95),
    ):
        "Group together a `model`, some `dls` and a `loss_func` to handle training"
        path = Path(path) if path is not None else getattr(dls, "path", Path("."))
        if loss_func is None:
            loss_func = getattr(dls.train_ds, "loss_func", None)
            assert (
                loss_func is not None
            ), "Could not infer loss function from the data, please pass a loss function."
        self.dls, self.model = dls, model
        store_attr(but="dls,model,cbs")
        self.training, self.create_mbar, self.logger, self.opt, self.cbs = (
            False,
            True,
            print,
            None,
            L(),
        )
        self.add_cbs(L(defaults.callbacks) + L(cbs))
        self("after_create")

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, v):
        self._metrics = L(v).map(mk_metric)

    def _grab_cbs(self, cb_cls):
        return L(cb for cb in self.cbs if isinstance(cb, cb_cls))

    def add_cbs(self, cbs):
        "Add `cbs` to the list of `Callback` and register `self` as their learner"
        L(cbs).map(self.add_cb)
        return self

    def remove_cbs(self, cbs):
        "Remove `cbs` from the list of `Callback` and deregister `self` as their learner"
        L(cbs).map(self.remove_cb)
        return self

    def add_cb(self, cb):
        "Add `cb` to the list of `Callback` and register `self` as their learner"
        if isinstance(cb, type):
            cb = cb()
        cb.learn = self
        setattr(self, cb.name, cb)
        self.cbs.append(cb)
        return self

    def remove_cb(self, cb):
        "Add `cb` from the list of `Callback` and deregister `self` as their learner"
        if isinstance(cb, type):
            self.remove_cbs(self._grab_cbs(cb))
        else:
            cb.learn = None
            if hasattr(self, cb.name):
                delattr(self, cb.name)
            if cb in self.cbs:
                self.cbs.remove(cb)
        return self

    @contextmanager
    def added_cbs(self, cbs):
        "Context manage that temporarily adds `cbs`"
        self.add_cbs(cbs)
        try:
            yield
        finally:
            self.remove_cbs(cbs)

    @contextmanager
    def removed_cbs(self, cbs):
        "Context manage that temporarily removes `cbs`"
        self.remove_cbs(cbs)
        try:
            yield self
        finally:
            self.add_cbs(cbs)

    def ordered_cbs(self, event):
        "List of `Callback`s, in order, for an `event` in the training loop"
        return [cb for cb in self.cbs.sorted("order") if hasattr(cb, event)]

    def __call__(self, event_name):
        "Call `event_name` for all `Callback`s in `self.cbs`"
        L(event_name).map(self._call_one)

    def _call_one(self, event_name):
        if not hasattr(event, event_name):
            raise Exception(f"missing {event_name}")
        for cb in self.cbs.sorted("order"):
            cb(event_name)

    def _bn_bias_state(self, with_bias):
        return norm_bias_params(self.model, with_bias).map(self.opt.state)

    def create_opt(self):
        "Create an optimizer with default hyper-parameters"
        self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)
        if not self.wd_bn_bias:
            for p in self._bn_bias_state(True):
                p["do_wd"] = False
        if self.train_bn:
            for p in self._bn_bias_state(False):
                p["force_train"] = True

    def _split(self, b):
        i = getattr(self.dls, "n_inp", 1 if len(b) == 1 else len(b) - 1)
        self.xb, self.yb = b[:i], b[i:]

    def _with_events(self, f, event_type, ex, final=noop):
        try:
            self(f"before_{event_type}")
            f()
        except ex:
            self(f"after_cancel_{event_type}")
        self(f"after_{event_type}")
        final()

    def all_batches(self):
        "Train or evaluate `self.model` on all the batches of `self.dl`"
        self.n_iter = len(self.dl)
        for o in enumerate(self.dl):
            self.one_batch(*o)

    def _do_one_batch(self):
        self.pred = self.model(*self.xb)
        self("after_pred")
        if len(self.yb):
            self.loss_grad = self.loss_func(self.pred, *self.yb)
            self.loss = self.loss_grad.clone()
        self("after_loss")
        if not self.training or not len(self.yb):
            return
        self("before_backward")
        self.loss_grad.backward()
        self._with_events(self.opt.step, "step", CancelStepException)
        self.opt.zero_grad()

    def one_batch(self, i, b):
        "Train or evaluate `self.model` on batch `(xb,yb)`"
        self.iter = i
        self._split(b)
        self._with_events(self._do_one_batch, "batch", CancelBatchException)

    def _do_epoch_train(self):
        self.dl = self.dls.train
        self._with_events(self.all_batches, "train", CancelTrainException)

    def _do_epoch_validate(self, ds_idx=1, dl=None):
        if dl is None:
            dl = self.dls[ds_idx]
        self.dl = dl
        with torch.no_grad():
            self._with_events(self.all_batches, "validate", CancelValidException)

    def _do_epoch(self):
        self._do_epoch_train()
        self._do_epoch_validate()

    def _do_fit(self):
        for epoch in range(self.n_epoch):
            self.epoch = epoch
            self._with_events(self._do_epoch, "epoch", CancelEpochException)

    def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False):
        "Fit `self.model` for `n_epoch` using `cbs`. Optionally `reset_opt`."
        with self.added_cbs(cbs):
            if reset_opt or not self.opt:
                self.create_opt()
            if wd is None:
                wd = self.wd
            if wd is not None:
                self.opt.set_hypers(wd=wd)
            self.opt.set_hypers(lr=self.lr if lr is None else lr)
            self.n_epoch = n_epoch
            self._with_events(
                self._do_fit, "fit", CancelFitException, self._end_cleanup
            )

    def _end_cleanup(self):
        self.dl, self.xb, self.yb, self.pred, self.loss = (
            None,
            (None,),
            (None,),
            None,
            None,
        )

    def __enter__(self):
        self(_before_epoch)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self(_after_epoch)

    def validation_context(self, cbs=None, inner=False):
        "A `ContextManagers` suitable for validation, with optional `cbs`"
        cms = [self.no_logging(), self.no_mbar()]
        if cbs:
            cms.append(self.added_cbs(cbs))
        if not inner:
            cms.append(self)
        return ContextManagers(cms)

    def validate(self, ds_idx=1, dl=None, cbs=None):
        "Validate on `dl` with potential new `cbs`."
        if dl is None:
            dl = self.dls[ds_idx]
        with self.validation_context(cbs=cbs):
            self._do_epoch_validate(ds_idx, dl)
        return getattr(self, "final_record", None)

    @delegates(GatherPredsCallback.__init__)
    def get_preds(
        self,
        ds_idx=1,
        dl=None,
        with_input=False,
        with_decoded=False,
        with_loss=False,
        act=None,
        inner=False,
        reorder=False,
        cbs=None,
        **kwargs,
    ):
        "Get the predictions and targets on the `ds_idx`-th dbunchset or `dl`, optionally `with_input` and `with_loss`"
        if dl is None:
            dl = self.dls[ds_idx]
        else:
            try:
                len(dl)
            except TypeError as e:
                raise TypeError(
                    "`dl` is something other than a single `DataLoader` object"
                )
        if reorder and hasattr(dl, "get_idxs"):
            raise NotImplementedError(
                "You're trying to use non-basic fastai functionality. You should use the entire fastai API to get this feature"
            )
        cb = GatherPredsCallback(with_input=with_input, with_loss=with_loss, **kwargs)
        ctx_mgrs = self.validation_context(cbs=L(cbs) + [cb], inner=inner)
        if with_loss:
            ctx_mgrs.append(self.loss_not_reduced())
        with ContextManagers(ctx_mgrs):
            self._do_epoch_validate(dl=dl)
            if act is None:
                act = getattr(self.loss_func, "activation", noop)
            res = cb.all_tensors()
            pred_i = 1 if with_input else 0
            if res[pred_i] is not None:
                res[pred_i] = act(res[pred_i])
                if with_decoded:
                    res.insert(
                        pred_i + 2,
                        getattr(self.loss_func, "decodes", noop)(res[pred_i]),
                    )
            if reorder and hasattr(dl, "get_idxs"):
                res = nested_reorder(res, tensor(idxs).argsort())
            return tuple(res)
        self._end_cleanup()

    def predict(self, item, rm_type_tfms=None, with_input=False):
        "Prediction on `item`, fully decoded, loss function decoded and probabilities"
        raise NotImplementedError(
            "You're trying to use non-basic fastai functionality. You should use the entire fastai API to get this feature"
        )

    def show_results(self, ds_idx=1, dl=None, max_n=9, shuffle=True, **kwargs):
        "Show some predictions on `ds_idx`-th dataset or `dl`"
        raise NotImplementedError(
            "You're trying to use non-basic fastai functionality. You should use the entire fastai API to get this feature"
        )

    def show_training_loop(self):
        "Show each step in the training loop"
        indent = 0
        for s in _loop:
            if s.startswith("Start"):
                print(f'{" "*indent}{s}')
                indent += 2
            elif s.startswith("End"):
                indent -= 2
                print(f'{" "*indent}{s}')
            else:
                print(f'{" "*indent} - {s:15}:', self.ordered_cbs(s))

    @contextmanager
    def no_logging(self):
        "Context manager to temporarily remove `logger`"
        return replacing_yield(self, "logger", noop)

    @contextmanager
    def no_mbar(self):
        "Context manager to temporarily prevent the master progress bar from being created"
        return replacing_yield(self, "create_mbar", False)

    @contextmanager
    def loss_not_reduced(self):
        "A context manager to evaluate `loss_func` with reduction set to none."
        if hasattr(self.loss_func, "reduction"):
            return replacing_yield(self.loss_func, "reduction", "none")
        else:
            return replacing_yield(
                self, "loss_func", partial(self.loss_func, reduction="none")
            )

    def to_detach(self, b, cpu=True, gather=True):
        "Calls `to_detach` if `self.dl` provides a `.to_detach` function otherwise calls global `to_detach`"
        return (
            self.dl.to_detach(b, cpu, gather)
            if hasattr(getattr(self, "dl", None), "to_detach")
            else to_detach(b, cpu, gather)
        )


Learner.x, Learner.y = add_props(lambda i, x: detuplify((x.xb, x.yb)[i]))

# Cell
if not hasattr(defaults, "callbacks"):
    defaults.callbacks = [TrainEvalCallback]

# Cell
def _before_batch_cb(f, self):
    xb, yb = f(self, self.xb, self.yb)
    self.learn.xb, self.learn.yb = xb, yb


# Cell
def before_batch_cb(f):
    "Shortcut for creating a Callback on the `before_batch` event, which takes and returns `xb,yb`"
    return Callback(before_batch=partial(_before_batch_cb, f))


# Cell
@patch
@delegates(save_model)
def save(self: Learner, file, **kwargs):
    "Save model and optimizer state (if `with_opt`) to `self.path/self.model_dir/file`"
    file = join_path_file(file, self.path / self.model_dir, ext=".pth")
    save_model(file, self.model, getattr(self, "opt", None), **kwargs)
    return file


# Cell
@patch
@delegates(load_model)
def load(self: Learner, file, device=None, **kwargs):
    "Load model and optimizer state (if `with_opt`) from `self.path/self.model_dir/file` using `device`"
    if device is None and hasattr(self.dls, "device"):
        device = self.dls.device
    if self.opt is None:
        self.create_opt()
    file = join_path_file(file, self.path / self.model_dir, ext=".pth")
    load_model(file, self.model, self.opt, device=device, **kwargs)
    return self


# Cell
@patch
def export(self: Learner, fname="export.pkl", pickle_module=pickle, pickle_protocol=2):
    "Export the content of `self` without the items and the optimizer state for inference"
    if rank_distrib():
        return  # don't export if child proc
    self._end_cleanup()
    old_dbunch = self.dls
    self.dls = self.dls.new_empty()
    state = self.opt.state_dict() if self.opt is not None else None
    self.opt = None
    with warnings.catch_warnings():
        # To avoid the warning that come from PyTorch about model not being checked
        warnings.simplefilter("ignore")
        torch.save(
            self,
            self.path / fname,
            pickle_module=pickle_module,
            pickle_protocol=pickle_protocol,
        )
    self.create_opt()
    if state is not None:
        self.opt.load_state_dict(state)
    self.dls = old_dbunch


# Cell
def load_learner(fname, cpu=True, pickle_module=pickle):
    "Load a `Learner` object in `fname`, optionally putting it on the `cpu`"
    distrib_barrier()
    res = torch.load(
        fname, map_location="cpu" if cpu else None, pickle_module=pickle_module
    )
    if hasattr(res, "to_fp32"):
        res = res.to_fp32()
    return res


# Cell
def to_detach_from_dl(
    learn: (Learner, NoneType), b: object, cpu: bool = True, gather: bool = True
):
    return (
        learn.dl.to_detach(b, cpu, gather)
        if hasattr(getattr(learn, "dl", None), "to_detach")
        else to_detach(b, cpu, gather)
    )


# Cell
class Metric:
    "Blueprint for defining a metric"

    def reset(self):
        "Reset inner state to prepare for new computation"
        pass

    def accumulate(self, learn):
        "Use `learn` to update the state with new results"
        pass

    @property
    def value(self):
        "The value of the metric"
        raise NotImplementedError

    @property
    def name(self):
        "Name of the `Metric`, camel-cased and with Metric removed"
        return class2attr(self, "Metric")


# Cell
def _maybe_reduce(val):
    if num_distrib() > 1:
        val = val.clone()
        torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.SUM)
        val /= num_distrib()
    return val


# Cell
class AvgMetric(Metric):
    "Average the values of `func` taking into account potential different batch sizes"

    def __init__(self, func):
        self.func = func

    def reset(self):
        self.total, self.count = 0.0, 0

    def accumulate(self, learn):
        bs = find_bs(learn.yb)
        self.total += learn.to_detach(self.func(learn.pred, *learn.yb)) * bs
        self.count += bs

    @property
    def value(self):
        return self.total / self.count if self.count != 0 else None

    @property
    def name(self):
        return (
            self.func.func.__name__
            if hasattr(self.func, "func")
            else self.func.__name__
        )


# Cell
class AvgLoss(Metric):
    "Average the losses taking into account potential different batch sizes"

    def reset(self):
        self.total, self.count = 0.0, 0

    def accumulate(self, learn):
        bs = find_bs(learn.yb)
        self.total += learn.to_detach(learn.loss.mean()) * bs
        self.count += bs

    @property
    def value(self):
        return self.total / self.count if self.count != 0 else None

    @property
    def name(self):
        return "loss"


# Cell
class AvgSmoothLoss(Metric):
    "Smooth average of the losses (exponentially weighted with `beta`)"

    def __init__(self, beta=0.98):
        self.beta = beta

    def reset(self):
        self.count, self.val = 0, tensor(0.0)

    def accumulate(self, learn):
        self.count += 1
        self.val = torch.lerp(
            to_detach(learn.loss.mean(), gather=False), self.val, self.beta
        )

    @property
    def value(self):
        return self.val / (1 - self.beta ** self.count)


# Cell
class ValueMetric(Metric):
    "Use to include a pre-calculated metric value (for instance calculated in a `Callback`) and returned by `func`"

    def __init__(self, func, metric_name=None):
        store_attr("func, metric_name")

    @property
    def value(self):
        return self.func()

    @property
    def name(self):
        return self.metric_name if self.metric_name else self.func.__name__


# Cell
from fastprogress.fastprogress import format_time


# Cell
def _maybe_item(t):
    t = t.value
    try:
        return t.item()
    except:
        return t


# Cell
class Recorder(Callback):
    "Callback that registers statistics (lr, loss and metrics) during training"
    _stateattrs = ("lrs", "iters", "losses", "values")
    remove_on_fetch, order = True, 50

    def __init__(
        self, add_time=True, train_metrics=False, valid_metrics=True, beta=0.98
    ):
        store_attr("add_time,train_metrics,valid_metrics")
        self.loss, self.smooth_loss = AvgLoss(), AvgSmoothLoss(beta=beta)

    def before_fit(self):
        "Prepare state for training"
        self.lrs, self.iters, self.losses, self.values = [], [], [], []
        names = self.metrics.attrgot("name")
        if self.train_metrics and self.valid_metrics:
            names = L("loss") + names
            names = names.map("train_{}") + names.map("valid_{}")
        elif self.valid_metrics:
            names = L("train_loss", "valid_loss") + names
        else:
            names = L("train_loss") + names
        if self.add_time:
            names.append("time")
        self.metric_names = "epoch" + names
        self.smooth_loss.reset()

    def after_batch(self):
        "Update all metrics and records lr and smooth loss in training"
        if len(self.yb) == 0:
            return
        mets = self._train_mets if self.training else self._valid_mets
        for met in mets:
            met.accumulate(self.learn)
        if not self.training:
            return
        self.lrs.append(self.opt.hypers[-1]["lr"])
        self.losses.append(self.smooth_loss.value)
        self.learn.smooth_loss = self.smooth_loss.value

    def before_epoch(self):
        "Set timer if `self.add_time=True`"
        self.cancel_train, self.cancel_valid = False, False
        if self.add_time:
            self.start_epoch = time.time()
        self.log = L(getattr(self, "epoch", 0))

    def before_train(self):
        "Reset loss and metrics state"
        self._train_mets[1:].map(Self.reset())

    def before_validate(self):
        "Reset loss and metrics state"
        self._valid_mets.map(Self.reset())

    def after_train(self):
        "Log loss and metric values on the training set (if `self.training_metrics=True`)"
        self.log += self._train_mets.map(_maybe_item)

    def after_validate(self):
        "Log loss and metric values on the validation set"
        self.log += self._valid_mets.map(_maybe_item)

    def after_cancel_train(self):
        "Ignore training metrics for this epoch"
        self.cancel_train = True

    def after_cancel_validate(self):
        "Ignore validation metrics for this epoch"
        self.cancel_valid = True

    def after_epoch(self):
        "Store and log the loss/metric values"
        self.learn.final_record = self.log[1:].copy()
        self.values.append(self.learn.final_record)
        if self.add_time:
            self.log.append(format_time(time.time() - self.start_epoch))
        self.logger(self.log)
        self.iters.append(self.smooth_loss.count)

    @property
    def _train_mets(self):
        if getattr(self, "cancel_train", False):
            return L()
        return L(self.smooth_loss) + (self.metrics if self.train_metrics else L())

    @property
    def _valid_mets(self):
        if getattr(self, "cancel_valid", False):
            return L()
        return L(self.loss) + self.metrics if self.valid_metrics else L()

    def plot_loss(self, skip_start=5, with_valid=True):
        "Plot the losses from `skip_start` and onward"
        plt.plot(
            list(range(skip_start, len(self.losses))),
            self.losses[skip_start:],
            label="train",
        )
        if with_valid:
            idx = (np.array(self.iters) < skip_start).sum()
            plt.plot(self.iters[idx:], L(self.values[idx:]).itemgot(1), label="valid")
            plt.legend()


# Cell
if Recorder not in defaults.callbacks:
    defaults.callbacks.append(Recorder)

# Cell
@patch
def freeze_to(self: Learner, n):
    "Freeze parameter groups up to `n`"
    if self.opt is None:
        self.create_opt()
    self.opt.freeze_to(n)
    self.opt.clear_state()


@patch
def freeze(self: Learner):
    "Freeze up to last parameter group"
    self.freeze_to(-1)


@patch
def unfreeze(self: Learner):
    "Unfreeze the entire model"
    self.freeze_to(0)


# Cell
@patch
def tta(
    self: Learner,
    ds_idx=1,
    dl=None,
    n=4,
    item_tfms=None,
    batch_tfms=None,
    beta=0.25,
    use_max=False,
):
    "Return predictions on the `ds_idx` dataset or `dl` using Test Time Augmentation"
    raise NotImplementedError(
        "You're trying to use non-basic fastai functionality. You should use the entire fastai API to get this feature"
    )
