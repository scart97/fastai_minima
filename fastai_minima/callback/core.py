# flake8: noqa

__all__ = [
    "CancelStepException",
    "CancelFitException",
    "CancelEpochException",
    "CancelTrainException",
    "CancelValidException",
    "CancelBatchException",
    "event",
    "Callback",
    "TrainEvalCallback",
    "GatherPredsCallback",
    "FetchPredsCallback",
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

from warnings import warn

# Cell
from fastcore.basics import (
    GetAttr,
    Stateful,
    class2attr,
    detuplify,
    mk_class,
    store_attr,
)
from fastcore.foundation import L
from fastcore.meta import funcs_kwargs

from ..utils import defaults, find_bs, noop, tensor, to_concat

# Cell
# nbdev_comment _all_ = ['CancelStepException','CancelFitException','CancelEpochException','CancelTrainException','CancelValidException','CancelBatchException']

# Cell
_events = L.split(
    "after_create before_fit before_epoch before_train before_batch after_pred after_loss \
    before_backward before_step after_cancel_step after_step after_cancel_batch after_batch after_cancel_train \
    after_train before_validate after_cancel_validate after_validate after_cancel_epoch \
    after_epoch after_cancel_fit after_fit"
)

mk_class(
    "event",
    **_events.map_dict(),
    doc="All possible events as attributes to get tab-completion and typo-proofing",
)

event.__module__ = "fastai_minima.callback"

# Cell
# nbdev_comment _all_ = ['event']

# Cell
_inner_loop = "before_batch after_pred after_loss before_backward before_step after_step after_cancel_batch after_batch".split()

# Cell
@funcs_kwargs(as_method=True)
class Callback(Stateful, GetAttr):
    "Basic class handling tweaks of the training loop by changing a `Learner` in various events"
    order, _default, learn, run, run_train, run_valid = (
        0,
        "learn",
        None,
        True,
        True,
        True,
    )
    _methods = _events

    def __init__(self, **kwargs):
        assert not kwargs, f"Passed unknown events: {kwargs}"

    def __repr__(self):
        return type(self).__name__

    def __call__(self, event_name):
        "Call `self.{event_name}` if it's defined"
        _run = (
            event_name not in _inner_loop
            or (self.run_train and getattr(self, "training", True))
            or (self.run_valid and not getattr(self, "training", False))
        )
        res = None
        if self.run and _run:
            res = getattr(self, event_name, noop)()
        if event_name == "after_fit":
            self.run = True  # Reset self.run to True at each end of fit
        return res

    def __setattr__(self, name, value):
        if hasattr(self.learn, name):
            warn(
                f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this"
            )
        super().__setattr__(name, value)

    @property
    def name(self):
        "Name of the `Callback`, camel-cased and with '*Callback*' removed"
        return class2attr(self, "Callback")


# Cell
class TrainEvalCallback(Callback):
    "`Callback` that tracks the number of iterations done and properly sets training/eval mode"
    order, run_valid = -10, False

    def after_create(self):
        self.learn.n_epoch = 1

    def before_fit(self):
        "Set the iter and epoch counters to 0, put the model and the right device"
        self.learn.epoch, self.learn.loss = 0, tensor(0.0)
        self.learn.train_iter, self.learn.pct_train = 0, 0.0
        if hasattr(self.dls, "device"):
            self.model.to(self.dls.device)
        if hasattr(self.model, "reset"):
            self.model.reset()

    def after_batch(self):
        "Update the iter counter (in training mode)"
        self.learn.pct_train += 1.0 / (self.n_iter * self.n_epoch)
        self.learn.train_iter += 1

    def before_train(self):
        "Set the model in training mode"
        self.learn.pct_train = self.epoch / self.n_epoch
        self.model.train()
        self.learn.training = True

    def before_validate(self):
        "Set the model in validation mode"
        self.model.eval()
        self.learn.training = False


# Cell
if not hasattr(defaults, "callbacks"):
    defaults.callbacks = [TrainEvalCallback]

# Cell
_ex_docs = dict(
    CancelBatchException="Skip the rest of this batch and go to `after_batch`",
    CancelTrainException="Skip the rest of the training part of the epoch and go to `after_train`",
    CancelValidException="Skip the rest of the validation part of the epoch and go to `after_validate`",
    CancelEpochException="Skip the rest of this epoch and go to `after_epoch`",
    CancelStepException="Skip stepping the optimizer",
    CancelFitException="Interrupts training and go to `after_fit`",
)

for c, d in _ex_docs.items():
    mk_class(c, sup=Exception, doc=d)

# Cell
# TODO: save_targs and save_preds only handle preds/targets that have one tensor, not tuples of tensors.
class GatherPredsCallback(Callback):
    "`Callback` that saves the predictions and targets, optionally `with_loss`"
    _stateattrs = ("preds", "targets", "inputs", "losses")

    def __init__(
        self,
        with_input=False,
        with_loss=False,
        save_preds=None,
        save_targs=None,
        concat_dim=0,
    ):
        store_attr("with_input,with_loss,save_preds,save_targs,concat_dim")

    def before_batch(self):
        if self.with_input:
            self.inputs.append((self.learn.to_detach(self.xb)))

    def before_validate(self):
        "Initialize containers"
        self.preds, self.targets = [], []
        if self.with_input:
            self.inputs = []
        if self.with_loss:
            self.losses = []

    def after_batch(self):
        "Save predictions, targets and potentially losses"
        if not hasattr(self, "pred"):
            return
        preds, targs = self.learn.to_detach(self.pred), self.learn.to_detach(self.yb)
        if self.save_preds is None:
            self.preds.append(preds)
        else:
            (self.save_preds / str(self.iter)).save_array(preds)
        if self.save_targs is None:
            self.targets.append(targs)
        else:
            (self.save_targs / str(self.iter)).save_array(targs[0])
        if self.with_loss:
            bs = find_bs(self.yb)
            loss = (
                self.loss if self.loss.numel() == bs else self.loss.view(bs, -1).mean(1)
            )
            self.losses.append(self.learn.to_detach(loss))

    def after_validate(self):
        "Concatenate all recorded tensors"
        if not hasattr(self, "preds"):
            return
        if self.with_input:
            self.inputs = detuplify(to_concat(self.inputs, dim=self.concat_dim))
        if not self.save_preds:
            self.preds = detuplify(to_concat(self.preds, dim=self.concat_dim))
        if not self.save_targs:
            self.targets = detuplify(to_concat(self.targets, dim=self.concat_dim))
        if self.with_loss:
            self.losses = to_concat(self.losses)

    def all_tensors(self):
        res = [
            None if self.save_preds else self.preds,
            None if self.save_targs else self.targets,
        ]
        if self.with_input:
            res = [self.inputs] + res
        if self.with_loss:
            res.append(self.losses)
        return res


# Cell
class FetchPredsCallback(Callback):
    "A callback to fetch predictions during the training loop"
    remove_on_fetch = True

    def __init__(
        self,
        ds_idx=1,
        dl=None,
        with_input=False,
        with_decoded=False,
        cbs=None,
        reorder=True,
    ):
        self.cbs = L(cbs)
        store_attr("ds_idx,dl,with_input,with_decoded,reorder")

    def after_validate(self):
        to_rm = L(cb for cb in self.learn.cbs if getattr(cb, "remove_on_fetch", False))
        with self.learn.removed_cbs(to_rm + self.cbs) as learn:
            self.preds = learn.get_preds(
                ds_idx=self.ds_idx,
                dl=self.dl,
                with_input=self.with_input,
                with_decoded=self.with_decoded,
                inner=True,
                reorder=self.reorder,
            )
