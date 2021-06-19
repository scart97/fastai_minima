import numpy as np
import torch
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis_gufunc.gufunc import gufunc_args

from fastai_minima.utils import to_concat


def torchify(args):
    args = tuple(torch.tensor(X) for X in args)
    return args


@given(
    gufunc_args("(a,b),(a,b),(a,b)->(c,b)", dtype=np.float_, elements=floats()).map(
        torchify
    )
)
def test_concat_tuple(x):
    out = to_concat(x)
    assert out.size(0) == 3 * x[0].size(0)
    assert out.size(1) == x[0].size(1)


@given(
    gufunc_args("(a,b),(a,b),(a,b)->(c,b)", dtype=np.float_, elements=floats()).map(
        torchify
    )
)
def test_concat_list(x):
    out = to_concat(list(list(x)))
    assert out.size(0) == 3 * x[0].size(0)
    assert out.size(1) == x[0].size(1)


@given(
    gufunc_args("(a,b),(a,b)->(a,c)", dtype=np.float_, elements=floats()).map(torchify)
)
def test_concat_second_dim(x):
    out = to_concat(x, dim=1)
    assert out.size(0) == x[0].size(0)
    assert out.size(1) == 2 * x[0].size(1)


def test_concat_none():
    assert to_concat(None) is None
