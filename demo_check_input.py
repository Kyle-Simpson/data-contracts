import xarray as xr

from src.xarray_annotations.lib import validate
from src.xarray_annotations.lib import schema

EXERCISE_COORD_ERROR = False
EXERCISE_DIM_ERROR = False

EXERCISE_MIN_VALUE_ERROR = False
EXERCISE_MAX_VALUE_ERROR = False

FOO = [1, 2]
BAR = [3, 4]
MIN_VAL = -100
MAX_VAL = 100

main_schema = schema.Schema(
    expected_dims=["foo", "bar"],
    dim_match_type="exact",
    expected_coords={"foo": FOO, "bar": BAR},
    coord_match_type="exact",
    minimum_value=MIN_VAL,
    maximum_value=MAX_VAL,
)


def setup() -> xr.DataArray:
    val_to_fill = 1
    if EXERCISE_MIN_VALUE_ERROR:
        val_to_fill = MIN_VAL - 1

    if EXERCISE_MAX_VALUE_ERROR:
        val_to_fill = MAX_VAL + 1

    data = xr.DataArray(
        data=[[val_to_fill, val_to_fill], [val_to_fill, val_to_fill]],
        dims=["foo", "bar"],
        coords={"foo": FOO, "bar": BAR}
    )

    if EXERCISE_COORD_ERROR:
        data = data.sel(foo=[1], drop=True)

    if EXERCISE_DIM_ERROR:
        data = data.sel(foo=[1]).squeeze("foo")
    
    return data


@validate.check_input(main_schema, arg_name="bar")
def main(foo: xr.DataArray, bar: xr.DataArray) -> None:
    pass


if __name__ == "__main__":
    input = setup()
    main(foo=input, bar=input)
