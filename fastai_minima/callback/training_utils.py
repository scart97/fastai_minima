# flake8: noqa

__all__ = ["ProgressCallback", "CollectDataCallback", "CudaCallback"]

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

from contextlib import contextmanager

from fastcore.basics import ifnone, patch
from fastprogress.fastprogress import master_bar, progress_bar

from ..learner import Learner
from ..utils import default_device, defaults, noop, to_device

# Cell
from .core import Callback


# Cell
class ProgressCallback(Callback):
    "A `Callback` to handle the display of progress bars"
    order, _stateattrs = 60, ("mbar", "pbar")

    def before_fit(self):
        "Setup the master bar over the epochs"
        assert hasattr(self.learn, "recorder")
        if self.create_mbar:
            self.mbar = master_bar(list(range(self.n_epoch)))
        if self.learn.logger != noop:
            self.old_logger, self.learn.logger = self.logger, self._write_stats
            self._write_stats(self.recorder.metric_names)
        else:
            self.old_logger = noop

    def before_epoch(self):
        "Update the master bar"
        if getattr(self, "mbar", False):
            self.mbar.update(self.epoch)

    def before_train(self):
        "Launch a progress bar over the training dataloader"
        self._launch_pbar()

    def before_validate(self):
        "Launch a progress bar over the validation dataloader"
        self._launch_pbar()

    def after_train(self):
        "Close the progress bar over the training dataloader"
        self.pbar.on_iter_end()

    def after_validate(self):
        "Close the progress bar over the validation dataloader"
        self.pbar.on_iter_end()

    def after_batch(self):
        "Update the current progress bar"
        self.pbar.update(self.iter + 1)
        if hasattr(self, "smooth_loss"):
            self.pbar.comment = f"{self.smooth_loss:.4f}"

    def _launch_pbar(self):
        self.pbar = progress_bar(
            self.dl, parent=getattr(self, "mbar", None), leave=False
        )
        self.pbar.update(0)

    def after_fit(self):
        "Close the master bar"
        if getattr(self, "mbar", False):
            self.mbar.on_iter_end()
            delattr(self, "mbar")
        if hasattr(self, "old_logger"):
            self.learn.logger = self.old_logger

    def _write_stats(self, log):
        if getattr(self, "mbar", False):
            self.mbar.write(
                [f"{l:.6f}" if isinstance(l, float) else str(l) for l in log],
                table=True,
            )


if not hasattr(defaults, "callbacks"):
    defaults.callbacks = [TrainEvalCallback, Recorder, ProgressCallback]
elif ProgressCallback not in defaults.callbacks:
    defaults.callbacks.append(ProgressCallback)

# Cell
@patch
@contextmanager
def no_bar(self: Learner):
    "Context manager that deactivates the use of progress bars"
    has_progress = hasattr(self, "progress")
    if has_progress:
        self.remove_cb(self.progress)
    try:
        yield self
    finally:
        if has_progress:
            self.add_cb(ProgressCallback())


# Cell
class CollectDataCallback(Callback):
    "Collect all batches, along with `pred` and `loss`, into `self.data`. Mainly for testing"

    def before_fit(self):
        self.data = L()

    def after_batch(self):
        self.data.append(self.learn.to_detach((self.xb, self.yb, self.pred, self.loss)))


# Cell
class CudaCallback(Callback):
    "Move data to CUDA device"

    def __init__(self, device=None):
        self.device = ifnone(device, default_device())

    def before_batch(self):
        self.learn.xb, self.learn.yb = to_device(self.xb), to_device(self.yb)

    def before_fit(self):
        self.model.to(self.device)
