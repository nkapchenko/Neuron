# -*- coding: utf-8 -*-

import pytest
from neuron.skeleton import fib

__author__ = "nkapchenko"
__copyright__ = "nkapchenko"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
