# flake8: noqa

__all__ = [
    "flatten_check",
    "AccumMetric",
    "skm_to_fastai",
    "optim_metric",
    "accuracy",
    "error_rate",
    "top_k_accuracy",
    "APScoreBinary",
    "BalancedAccuracy",
    "BrierScore",
    "CohenKappa",
    "F1Score",
    "FBeta",
    "HammingLoss",
    "Jaccard",
    "Precision",
    "Recall",
    "RocAuc",
    "RocAucBinary",
    "MatthewsCorrCoef",
    "Perplexity",
    "perplexity",
    "accuracy_multi",
    "APScoreMulti",
    "BrierScoreMulti",
    "F1ScoreMulti",
    "FBetaMulti",
    "HammingLossMulti",
    "JaccardMulti",
    "MatthewsCorrCoefMulti",
    "PrecisionMulti",
    "RecallMulti",
    "RocAucMulti",
    "mse",
    "rmse",
    "mae",
    "msle",
    "exp_rmspe",
    "ExplainedVariance",
    "R2Score",
    "PearsonCorrCoef",
    "SpearmanCorrCoef",
    "foreground_acc",
    "Dice",
    "DiceMulti",
    "JaccardCoeff",
    "CorpusBLEUMetric",
    "LossMetric",
    "LossMetrics",
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

from collections import Counter
from functools import partial

import numpy as np
import scipy.stats as scs

# Cell
import sklearn.metrics as skm
import torch
import torch.nn.functional as F
from fastcore.basics import mk_class, store_attr
from fastcore.meta import delegates

from .learner import AvgLoss, AvgMetric, Learner, Metric
from .utils import to_detach


# Cell
def flatten_check(inp, targ):
    "Check that `out` and `targ` have the same number of elements and flatten them."
    inp, targ = inp.contiguous().view(-1), targ.contiguous().view(-1)
    test_eq(len(inp), len(targ))
    return inp, targ


# Cell
mk_class(
    "ActivationType",
    **{o: o.lower() for o in ["No", "Sigmoid", "Softmax", "BinarySoftmax"]},
    doc="All possible activation classes for `AccumMetric",
)

# Cell
class AccumMetric(Metric):
    "Stores predictions and targets on CPU in accumulate to perform final calculations with `func`."

    def __init__(
        self,
        func,
        dim_argmax=None,
        activation=ActivationType.No,
        thresh=None,
        to_np=False,
        invert_arg=False,
        flatten=True,
        **kwargs,
    ):
        store_attr("func,dim_argmax,activation,thresh,flatten")
        self.to_np, self.invert_args, self.kwargs = to_np, invert_arg, kwargs

    def reset(self):
        "Clear all targs and preds"
        self.targs, self.preds = [], []

    def accumulate(self, learn: Learner):
        "Store targs and preds from `learn`, using activation function and argmax as appropriate"
        pred = learn.pred
        if self.activation in [ActivationType.Softmax, ActivationType.BinarySoftmax]:
            pred = F.softmax(pred, dim=self.dim_argmax)
            if self.activation == ActivationType.BinarySoftmax:
                pred = pred[:, -1]
        elif self.activation == ActivationType.Sigmoid:
            pred = torch.sigmoid(pred)
        elif self.dim_argmax:
            pred = pred.argmax(dim=self.dim_argmax)
        if self.thresh:
            pred = pred >= self.thresh
        self.accum_values(pred, learn.y, learn)

    def accum_values(self, preds, targs, learn=None):
        "Store targs and preds"
        to_d = learn.to_detach if learn is not None else to_detach
        preds, targs = to_d(preds), to_d(targs)
        if self.flatten:
            preds, targs = flatten_check(preds, targs)
        self.preds.append(preds)
        self.targs.append(targs)

    def __call__(self, preds, targs):
        "Calculate metric on one batch of data"
        self.reset()
        self.accum_values(preds, targs)
        return self.value

    @property
    def value(self):
        "Value of the metric using accumulated preds and targs"
        if len(self.preds) == 0:
            return
        preds, targs = torch.cat(self.preds), torch.cat(self.targs)
        if self.to_np:
            preds, targs = preds.numpy(), targs.numpy()
        return (
            self.func(targs, preds, **self.kwargs)
            if self.invert_args
            else self.func(preds, targs, **self.kwargs)
        )

    @property
    def name(self):
        return (
            self.func.func.__name__
            if hasattr(self.func, "func")
            else self.func.__name__
        )


# Cell
def skm_to_fastai(func, is_class=True, thresh=None, axis=-1, activation=None, **kwargs):
    "Convert `func` from sklearn.metrics to a fastai metric"
    dim_argmax = axis if is_class and thresh is None else None
    if activation is None:
        activation = (
            ActivationType.Sigmoid
            if (is_class and thresh is not None)
            else ActivationType.No
        )
    return AccumMetric(
        func,
        dim_argmax=dim_argmax,
        activation=activation,
        thresh=thresh,
        to_np=True,
        invert_arg=True,
        **kwargs,
    )


# Cell
def optim_metric(f, argname, bounds, tol=0.01, do_neg=True, get_x=False):
    "Replace metric `f` with a version that optimizes argument `argname`"

    def _f(preds, targs):
        def minfunc(x):
            kwargs = {argname: x}
            res = f(preds, targs, **kwargs)
            return -res if do_neg else res

        optres = scipy.optimize.minimize_scalar(
            minfunc, bounds=bounds, method="bounded", options={"xatol": 0.01}
        )
        fun = -optres.fun if do_neg else optres.fun
        return (fun, optres.x) if get_x else fun

    _f.__name__ = f"opt_{f.__name__}"
    return _f


# Cell
def accuracy(inp, targ, axis=-1):
    "Compute accuracy with `targ` when `pred` is bs * n_classes"
    pred, targ = flatten_check(inp.argmax(dim=axis), targ)
    return (pred == targ).float().mean()


# Cell
def error_rate(inp, targ, axis=-1):
    "1 - `accuracy`"
    return 1 - accuracy(inp, targ, axis=axis)


# Cell
def top_k_accuracy(inp, targ, k=5, axis=-1):
    "Computes the Top-k accuracy (`targ` is in the top `k` predictions of `inp`)"
    inp = inp.topk(k=k, dim=axis)[1]
    targ = targ.unsqueeze(dim=axis).expand_as(inp)
    return (inp == targ).sum(dim=-1).float().mean()


# Cell
def APScoreBinary(axis=-1, average="macro", pos_label=1, sample_weight=None):
    "Average Precision for single-label binary classification problems"
    return skm_to_fastai(
        skm.average_precision_score,
        axis=axis,
        activation=ActivationType.BinarySoftmax,
        average=average,
        pos_label=pos_label,
        sample_weight=sample_weight,
    )


# Cell
def BalancedAccuracy(axis=-1, sample_weight=None, adjusted=False):
    "Balanced Accuracy for single-label binary classification problems"
    return skm_to_fastai(
        skm.balanced_accuracy_score,
        axis=axis,
        sample_weight=sample_weight,
        adjusted=adjusted,
    )


# Cell
def BrierScore(axis=-1, sample_weight=None, pos_label=None):
    "Brier score for single-label classification problems"
    return skm_to_fastai(
        skm.brier_score_loss,
        axis=axis,
        sample_weight=sample_weight,
        pos_label=pos_label,
    )


# Cell
def CohenKappa(axis=-1, labels=None, weights=None, sample_weight=None):
    "Cohen kappa for single-label classification problems"
    return skm_to_fastai(
        skm.cohen_kappa_score,
        axis=axis,
        labels=labels,
        weights=weights,
        sample_weight=sample_weight,
    )


# Cell
def F1Score(axis=-1, labels=None, pos_label=1, average="binary", sample_weight=None):
    "F1 score for single-label classification problems"
    return skm_to_fastai(
        skm.f1_score,
        axis=axis,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
    )


# Cell
def FBeta(
    beta, axis=-1, labels=None, pos_label=1, average="binary", sample_weight=None
):
    "FBeta score with `beta` for single-label classification problems"
    return skm_to_fastai(
        skm.fbeta_score,
        axis=axis,
        beta=beta,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
    )


# Cell
def HammingLoss(axis=-1, sample_weight=None):
    "Hamming loss for single-label classification problems"
    return skm_to_fastai(skm.hamming_loss, axis=axis, sample_weight=sample_weight)


# Cell
def Jaccard(axis=-1, labels=None, pos_label=1, average="binary", sample_weight=None):
    "Jaccard score for single-label classification problems"
    return skm_to_fastai(
        skm.jaccard_score,
        axis=axis,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
    )


# Cell
def Precision(axis=-1, labels=None, pos_label=1, average="binary", sample_weight=None):
    "Precision for single-label classification problems"
    return skm_to_fastai(
        skm.precision_score,
        axis=axis,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
    )


# Cell
def Recall(axis=-1, labels=None, pos_label=1, average="binary", sample_weight=None):
    "Recall for single-label classification problems"
    return skm_to_fastai(
        skm.recall_score,
        axis=axis,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
    )


# Cell
def RocAuc(
    axis=-1, average="macro", sample_weight=None, max_fpr=None, multi_class="ovr"
):
    "Area Under the Receiver Operating Characteristic Curve for single-label multiclass classification problems"
    assert multi_class in ["ovr", "ovo"]
    return skm_to_fastai(
        skm.roc_auc_score,
        axis=axis,
        activation=ActivationType.Softmax,
        flatten=False,
        average=average,
        sample_weight=sample_weight,
        max_fpr=max_fpr,
        multi_class=multi_class,
    )


# Cell
def RocAucBinary(
    axis=-1, average="macro", sample_weight=None, max_fpr=None, multi_class="raise"
):
    "Area Under the Receiver Operating Characteristic Curve for single-label binary classification problems"
    return skm_to_fastai(
        skm.roc_auc_score,
        axis=axis,
        activation=ActivationType.BinarySoftmax,
        average=average,
        sample_weight=sample_weight,
        max_fpr=max_fpr,
        multi_class=multi_class,
    )


# Cell
def MatthewsCorrCoef(sample_weight=None, **kwargs):
    "Matthews correlation coefficient for single-label classification problems"
    return skm_to_fastai(skm.matthews_corrcoef, sample_weight=sample_weight, **kwargs)


# Cell
class Perplexity(AvgLoss):
    "Perplexity (exponential of cross-entropy loss) for Language Models"

    @property
    def value(self):
        return torch.exp(self.total / self.count) if self.count != 0 else None

    @property
    def name(self):
        return "perplexity"


perplexity = Perplexity()

# Cell
def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    inp, targ = flatten_check(inp, targ)
    if sigmoid:
        inp = inp.sigmoid()
    return ((inp > thresh) == targ.bool()).float().mean()


# Cell
def APScoreMulti(sigmoid=True, average="macro", pos_label=1, sample_weight=None):
    "Average Precision for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(
        skm.average_precision_score,
        activation=activation,
        flatten=False,
        average=average,
        pos_label=pos_label,
        sample_weight=sample_weight,
    )


# Cell
def BrierScoreMulti(thresh=0.5, sigmoid=True, sample_weight=None, pos_label=None):
    "Brier score for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(
        skm.brier_score_loss,
        thresh=thresh,
        activation=activation,
        flatten=False,
        sample_weight=sample_weight,
        pos_label=pos_label,
    )


# Cell
def F1ScoreMulti(
    thresh=0.5,
    sigmoid=True,
    labels=None,
    pos_label=1,
    average="macro",
    sample_weight=None,
):
    "F1 score for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(
        skm.f1_score,
        thresh=thresh,
        activation=activation,
        flatten=False,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
    )


# Cell
def FBetaMulti(
    beta,
    thresh=0.5,
    sigmoid=True,
    labels=None,
    pos_label=1,
    average="macro",
    sample_weight=None,
):
    "FBeta score with `beta` for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(
        skm.fbeta_score,
        thresh=thresh,
        activation=activation,
        flatten=False,
        beta=beta,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
    )


# Cell
def HammingLossMulti(thresh=0.5, sigmoid=True, labels=None, sample_weight=None):
    "Hamming loss for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(
        skm.hamming_loss,
        thresh=thresh,
        activation=activation,
        flatten=False,
        sample_weight=sample_weight,
    )


# Cell
def JaccardMulti(
    thresh=0.5,
    sigmoid=True,
    labels=None,
    pos_label=1,
    average="macro",
    sample_weight=None,
):
    "Jaccard score for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(
        skm.jaccard_score,
        thresh=thresh,
        activation=activation,
        flatten=False,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
    )


# Cell
def MatthewsCorrCoefMulti(thresh=0.5, sigmoid=True, sample_weight=None):
    "Matthews correlation coefficient for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(
        skm.matthews_corrcoef,
        thresh=thresh,
        activation=activation,
        flatten=False,
        sample_weight=sample_weight,
    )


# Cell
def PrecisionMulti(
    thresh=0.5,
    sigmoid=True,
    labels=None,
    pos_label=1,
    average="macro",
    sample_weight=None,
):
    "Precision for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(
        skm.precision_score,
        thresh=thresh,
        activation=activation,
        flatten=False,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
    )


# Cell
def RecallMulti(
    thresh=0.5,
    sigmoid=True,
    labels=None,
    pos_label=1,
    average="macro",
    sample_weight=None,
):
    "Recall for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(
        skm.recall_score,
        thresh=thresh,
        activation=activation,
        flatten=False,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
    )


# Cell
def RocAucMulti(sigmoid=True, average="macro", sample_weight=None, max_fpr=None):
    "Area Under the Receiver Operating Characteristic Curve for multi-label binary classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(
        skm.roc_auc_score,
        activation=activation,
        flatten=False,
        average=average,
        sample_weight=sample_weight,
        max_fpr=max_fpr,
    )


# Cell
def mse(inp, targ):
    "Mean squared error between `inp` and `targ`."
    return F.mse_loss(*flatten_check(inp, targ))


# Cell
def _rmse(inp, targ):
    return torch.sqrt(F.mse_loss(inp, targ))


rmse = AccumMetric(_rmse)
rmse.__doc__ = "Root mean squared error"

# Cell
def mae(inp, targ):
    "Mean absolute error between `inp` and `targ`."
    inp, targ = flatten_check(inp, targ)
    return torch.abs(inp - targ).mean()


# Cell
def msle(inp, targ):
    "Mean squared logarithmic error between `inp` and `targ`."
    inp, targ = flatten_check(inp, targ)
    return F.mse_loss(torch.log(1 + inp), torch.log(1 + targ))


# Cell
def _exp_rmspe(inp, targ):
    inp, targ = torch.exp(inp), torch.exp(targ)
    return torch.sqrt(((targ - inp) / targ).pow(2).mean())


exp_rmspe = AccumMetric(_exp_rmspe)
exp_rmspe.__doc__ = (
    "Root mean square percentage error of the exponential of  predictions and targets"
)

# Cell
def ExplainedVariance(sample_weight=None):
    "Explained variance between predictions and targets"
    return skm_to_fastai(
        skm.explained_variance_score, is_class=False, sample_weight=sample_weight
    )


# Cell
def R2Score(sample_weight=None):
    "R2 score between predictions and targets"
    return skm_to_fastai(skm.r2_score, is_class=False, sample_weight=sample_weight)


# Cell
@delegates(AccumMetric)
def PearsonCorrCoef(dim_argmax=None, **kwargs):
    "Pearson correlation coefficient for regression problem"

    def pearsonr(x, y):
        return scs.pearsonr(x, y)[0]

    return AccumMetric(pearsonr, invert_arg=False, dim_argmax=dim_argmax, **kwargs)


# Cell
@delegates(AccumMetric)
def SpearmanCorrCoef(dim_argmax=None, axis=0, nan_policy="propagate", **kwargs):
    "Spearman correlation coefficient for regression problem"

    def spearmanr(a, b=None, **kwargs):
        return scs.spearmanr(a, b, **kwargs)[0]

    return AccumMetric(
        partial(spearmanr, axis=axis, nan_policy=nan_policy),
        invert_arg=False,
        dim_argmax=dim_argmax,
        **kwargs,
    )


# Cell
def foreground_acc(inp, targ, bkg_idx=0, axis=1):
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()


# Cell
class Dice(Metric):
    "Dice coefficient metric for binary target in segmentation"

    def __init__(self, axis=1):
        self.axis = axis

    def reset(self):
        self.inter, self.union = 0, 0

    def accumulate(self, learn):
        pred, targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.y)
        self.inter += (pred * targ).float().sum().item()
        self.union += (pred + targ).float().sum().item()

    @property
    def value(self):
        return 2.0 * self.inter / self.union if self.union > 0 else None


# Cell
class DiceMulti(Metric):
    "Averaged Dice metric (Macro F1) for multiclass target in segmentation"

    def __init__(self, axis=1):
        self.axis = axis

    def reset(self):
        self.inter, self.union = {}, {}

    def accumulate(self, learn):
        pred, targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.y)
        for c in range(learn.pred.shape[self.axis]):
            p = torch.where(pred == c, 1, 0)
            t = torch.where(targ == c, 1, 0)
            c_inter = (p * t).float().sum().item()
            c_union = (p + t).float().sum().item()
            if c in self.inter:
                self.inter[c] += c_inter
                self.union[c] += c_union
            else:
                self.inter[c] = c_inter
                self.union[c] = c_union

    @property
    def value(self):
        binary_dice_scores = np.array([])
        for c in self.inter:
            binary_dice_scores = np.append(
                binary_dice_scores,
                2.0 * self.inter[c] / self.union[c] if self.union[c] > 0 else np.nan,
            )
        return np.nanmean(binary_dice_scores)


# Cell
class JaccardCoeff(Dice):
    "Implementation of the Jaccard coefficient that is lighter in RAM"

    @property
    def value(self):
        return self.inter / (self.union - self.inter) if self.union > 0 else None


# Cell
class CorpusBLEUMetric(Metric):
    def __init__(self, vocab_sz=5000, axis=-1):
        "BLEU Metric calculated over the validation corpus"
        self.metric_name = "CorpusBLEU"
        self.axis, self.vocab_sz = axis, vocab_sz
        self.pred_len, self.targ_len, self.samp_idx, self.corrects, self.counts, = (
            0,
            0,
            0,
            [0] * 4,
            [0] * 4,
        )

    def reset(self):
        self.pred_len, self.targ_len, self.corrects, self.counts = (
            0,
            0,
            [0] * 4,
            [0] * 4,
        )

    class NGram:
        def __init__(self, ngram, max_n=5000):
            self.ngram, self.max_n = ngram, max_n

        def __eq__(self, other):
            if len(self.ngram) != len(other.ngram):
                return False
            return np.all(np.array(self.ngram) == np.array(other.ngram))

        def __hash__(self):
            return int(sum([o * self.max_n ** i for i, o in enumerate(self.ngram)]))

    def get_grams(self, x, n, max_n=5000):
        return (
            x
            if n == 1
            else [self.NGram(x[i : i + n], max_n=max_n) for i in range(len(x) - n + 1)]
        )

    def get_correct_ngrams(self, pred, targ, n, max_n=5000):
        pred_grams, targ_grams = self.get_grams(pred, n, max_n=max_n), self.get_grams(
            targ, n, max_n=max_n
        )
        pred_cnt, targ_cnt = Counter(pred_grams), Counter(targ_grams)
        return sum([min(c, targ_cnt[g]) for g, c in pred_cnt.items()]), len(pred_grams)

    def accumulate(self, learn):
        if learn.training:
            return None
        else:
            last_output = learn.pred.argmax(dim=self.axis)
            last_target = learn.y
            for pred, targ in zip(last_output.cpu().numpy(), last_target.cpu().numpy()):
                self.pred_len += len(pred)
                self.targ_len += len(targ)
                smooth_mteval = 1
                for i in range(4):
                    c, t = self.get_correct_ngrams(
                        pred, targ, i + 1, max_n=self.vocab_sz
                    )
                    if c == 0:
                        smooth_mteval *= 2
                        c = (
                            1 / smooth_mteval
                        )  # exp smoothing, method 3 from http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
                    self.corrects[i] += c
                    self.counts[i] += t

    @property
    def value(self):
        if self.counts == 0:
            return None
        elif max(self.corrects) == 0:
            return 0.0
        else:
            precs = [c / t for c, t in zip(self.corrects, self.counts)]
            len_penalty = (
                math.exp(1 - self.targ_len / self.pred_len)
                if self.pred_len < self.targ_len
                else 1
            )
            return len_penalty * ((precs[0] * precs[1] * precs[2] * precs[3]) ** 0.25)


# Cell
class LossMetric(AvgMetric):
    "Create a metric from `loss_func.attr` named `nm`"

    def __init__(self, attr, nm=None):
        store_attr("attr,nm")

    def accumulate(self, learn):
        bs = find_bs(learn.yb)
        self.total += learn.to_detach(getattr(learn.loss_func, self.attr, 0)) * bs
        self.count += bs

    @property
    def name(self):
        return self.attr if self.nm is None else self.nm


# Cell
def LossMetrics(attrs, nms=None):
    "List of `LossMetric` for each of `attrs` and `nms`"
    if isinstance(attrs, str):
        attrs = attrs.split(",")
    nms = attrs if nms is None else nms.split(",") if isinstance(nms, str) else nms
    return [LossMetric(a, n) for a, n in zip(attrs, nms)]
