from typing import Callable

import pytest
import xarray as xr

from src.xarray_annotations.lib import schema, validate

input_func_schema = schema.Schema(
    expected_dims=["age_group_id", "sex_id"],
)
output_func_schema = schema.Schema()

@validate.check_input(input_func_schema, arg_name="foo")
def input_func(foo: xr.DataArray, bar: str) -> None:
    pass


@pytest.mark.unit
class TestCheckInput:
    def test_exact_coord_error(self) -> None:
        """Ensure an error is raised when checking data with
        invalid coordinates.
        """
        # Set up schema and function
        my_schema = schema.Schema(
            expected_coords={"age_group_id": [1, 2], "sex_id": [2, 3]},
            coord_match_type="exact",
            raise_errors=True,
        )

        @validate.check_input(my_schema, arg_name="foo")
        def input_func(foo: xr.DataArray, bar: str) -> None:
            pass

        # Set up input data that will raise an error
        foo = xr.DataArray(
            data=[[1], [1]],
            dims=["age_group_id", "sex_id"],
            coords={"age_group_id": [1, 2], "sex_id": [2]},
        )

        # Exercise and Verify
        with pytest.raises(
            Exception, match="Data coords to not exactly match the expected ones"
        ):
            input_func(foo=foo, bar="something")

    def test_minimum_coord_error(self) -> None:
        """Ensure an error is raised when checking data with
        invalid coordinates.
        """
        # Set up schema and function
        my_schema = schema.Schema(
            expected_coords={"age_group_id": [1, 2], "sex_id": [2, 3]},
            coord_match_type="minimum",
            raise_errors=True,
        )

        @validate.check_input(my_schema, arg_name="foo")
        def input_func(foo: xr.DataArray, bar: str) -> None:
            pass

        # Set up input data that will raise an error
        foo = xr.DataArray(
            data=[[1], [1]],
            dims=["age_group_id", "sex_id"],
            coords={"age_group_id": [1, 2], "sex_id": [2]},
        )

        # Exercise and Verify
        with pytest.raises(
            Exception, match="Data has mismatching coordinate values"
        ):
            input_func(foo=foo, bar="something")

    def test_exact_dim_error(self) -> None:
        """Ensure an error is raised when checking data with
        invalid dims.
        """
        pass

    def test_minimum_dim_error(self) -> None:
        """Ensure an error is raised when checking data with
        invalid dims.
        """
        pass


@pytest.mark.unit
class TestCheckOutput:
    def test_exact_coord_error(self) -> None:
        """Ensure an error is raised when checking data with
        invalid coordinates.
        """
        pass

    def test_minimum_coord_error(self) -> None:
        """Ensure an error is raised when checking data with
        invalid coordinates.
        """
        pass

    def test_exact_dim_error(self) -> None:
        """Ensure an error is raised when checking data with
        invalid dims.
        """
        pass

    def test_minimum_dim_error(self) -> None:
        """Ensure an error is raised when checking data with
        invalid dims.
        """
        pass


@pytest.mark.unit
class TestDimCallback:
    pass


@pytest.mark.unit
class TestCoordCallback:
    pass
