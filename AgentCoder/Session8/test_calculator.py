# test_calculator.py
import pytest
from calculator import divide


def test_divide_normal():
    assert divide(6, 3) == 2.0


def test_divide_by_zero():
    # File test yêu cầu: khi chia cho 0 phải trả về 0.0 thay vì bị sập chương trình
    assert divide(5, 0) == 0.0