# global
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def custom_test_values(draw, *, values):
    return "int32", values

@handle_cmd_line_args
@given(
    dtype_and_x=custom_test_values(values=[
        [0, 1],
        [1, 2],
        [-1, 2],
        [3, 4],
        [3, 5],
        [-3, 4],
        [-3, 5],
    ]),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.core.multiarray.normalize_axis_index"
    ),
)
def test_numpy_normalize_axis_index(
    dtype_and_x,
    num_positional_args,
    fw,
):
    input_dtype, x = dtype_and_x
    x_axis = list(map(lambda x: x[0], x))
    x_ndims = list(map(lambda x: x[1], x))
    for i in range(len(x)):
        helpers.test_frontend_function(
            axis=x_axis[i],
            ndim=x_ndims[i],
            input_dtypes=input_dtype,
            as_variable_flags=False,
            native_array_flags=False,
            with_out=False,
            num_positional_args=num_positional_args,
            fw=fw,
            frontend="numpy",
            fn_tree="core.multiarray.normalize_axis_index",
        )
