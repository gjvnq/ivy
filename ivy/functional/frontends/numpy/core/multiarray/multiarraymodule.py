
# From the reference implementation, we discover that this function works on
# Python's `int` type. Thus I think that the type annotations here present are
# fitting.
#
# Reference implementation:
# - https://github.com/numpy/numpy/blob/v1.23.2/numpy/core/src/multiarray/multiarraymodule.c#L4219-L4240
# - https://github.com/numpy/numpy/blob/v1.23.2/numpy/core/src/multiarray/common.h#L129-L169
def normalize_axis_index(axis: int, ndim: int, /) -> int:
    assert ndim >= 0
    if 0 <= axis < ndim:
        return axis
    elif -ndim < axis <= 0:
        return ndim+axis
    else:
        raise ValueError(f"axis {repr(axis)} is out of bounds for array of dimension {repr(ndim)}")
