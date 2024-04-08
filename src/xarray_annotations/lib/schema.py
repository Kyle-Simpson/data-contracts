import logging
from typing import Any, Callable

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

EXACT_MATCH_TYPE = "exact"
MINIMUM_MATCH_TYPE = "minimum"
VALID_MATCH_TYPES = [EXACT_MATCH_TYPE, MINIMUM_MATCH_TYPE]


def coords_as_dict(data: xr.DataArray) -> dict[str, Any]:
    coord_dict = dict()
    for item, value in data.coords.items():
        if isinstance(value.values.tolist(), int):
            val_to_add = [value.values.item()]
        else:
            val_to_add = list(value.values)

        coord_dict[item] = val_to_add
    return coord_dict


class Schema:
    def __init__(
        self,
        expected_dims: list[str] | Callable = [],
        dim_match_type: str = EXACT_MATCH_TYPE,
        expected_coords: dict[str, Any] | Callable = dict(),
        coord_match_type: str = EXACT_MATCH_TYPE,
        minimum_value: float = -np.inf,
        maximum_value: float = np.inf,
        nan_values_allowed: bool = True,
        raise_errors: bool = True,
    ) -> None:
        """Object performing dimension, coordinate, and value validation.

        Args:
            expected_dims (list[str] | Callable): The expected dimensions
                for an xr.DataArray, or a callable that will return the
                expected dimensions.
            match_dim_type (str): The type of match to perform on dimensions.
            expected_coords (dict[str, Any] | Callable): The expected coordinates
                for an xr.DataArray, or a callable that will return the expected
                coordinates.
            match_coord_type (str): The type of match to perform on coordinates.
            minimum_value (float): The minimum acceptable value for the data.
            maximum_value (float): The maximum acceptable value for the data.
            nan_values_allowed (bool): Whether or not the data is allowed to have
                NaN values.
            raise_errors (bool): Whether or not errors should be raised when a
                validation fails
        """
        if dim_match_type not in VALID_MATCH_TYPES:
            raise Exception(
                f"Invalid match type: {dim_match_type}. Please supply one of the "
                f"following: {VALID_MATCH_TYPES}."
            )

        self._expected_dims = expected_dims
        self._expected_coords = expected_coords

        self._dim_match_type = dim_match_type
        self._coord_match_type = coord_match_type

        self._min_value = minimum_value
        self._max_value = maximum_value

        self._nan_values_allowed = nan_values_allowed

        self._raise_errors = raise_errors

    def resolve_expected_dims(self, **kwargs: Any) -> None:
        if callable(self._expected_dims):
            self._expected_dims = self._expected_dims(**kwargs)

    def resolve_expected_coords(self, **kwargs: Any) -> None:
        if callable(self._expected_coords):
            self._expected_coords = self._expected_coords(**kwargs)

    def _log_or_raise(self, message: str) -> None:
        if self._raise_errors:
            raise Exception(message)
        logger.warning(message)

    def _match_dims_exactly(self, data: xr.DataArray) -> None:
        if len(self._expected_dims) == 0:
            return

        if list(data.dims) != self._expected_dims:
            self._log_or_raise(
                f"Data dims do not exactly match the expected ones. "
                f"Expected: {self._expected_dims}. Found: {data.dims}."
            )

    def _match_minimum_dims(self, data: xr.DataArray) -> None:
        if len(self._expected_dims) == 0:
            return

        missing_dims = []
        for dim in self._expected_dims:
            if dim not in self._expected_dims:
                missing_dims.append(dim)
        
        if missing_dims:
            self._log_or_raise(
                f"Data dims do not contain the minimum expected ones. "
                f"Expected: {self._expected_dims}. Found: {data.dims}. "
                f"Missing: {missing_dims}."
            )

    def _match_coords_exactly(self, data: xr.DataArray) -> None:
        if len(self._expected_coords) == 0:
            return

        coords = coords_as_dict(data)

        if self._expected_coords.items() != coords.items():
            self._log_or_raise(
                f"Data coords to not exactly match the expected ones. "
                f"Expected: {self._expected_coords}. Found: {coords}."
            )

    def _match_minimum_coords(self, data: xr.DataArray) -> None:
        if len(self._expected_coords) == 0:
            return

        coords = coords_as_dict(data)

        missing_coords = []
        mismatching_coord_values = []
        for coord_name, expected_coord_values in self._expected_coords.items():
            if coord_name not in coords.keys():
                missing_coords.append(coord_name)

            if not set(coords[coord_name]).issuperset(set(expected_coord_values)):
                mismatching_coord_values.append(coord_name)

        if missing_coords:
            self._log_or_raise(
                f"Data is missing coordinate(s): {missing_coords}. "
                f"Expected: {self._expected_coords.keys()}. Found: {coords.keys()}"
            )

        if mismatching_coord_values:
            expected_str = ", ".join(
                [
                    f"{coord}: {self._expected_coords[coord]}"
                    for coord in mismatching_coord_values
                ]
            )
            found_str = ", ".join(
                [
                    f"{coord}: {vals}"
                    for coord, vals in coords.items()
                    if coord in mismatching_coord_values
                ]
            )
            self._log_or_raise(
                f"Data has mismatching coordinate values for coords: "
                f"{mismatching_coord_values}. Expected: {expected_str}. "
                f"Found: {found_str}."
            )

    def _ensure_values_above_minimum(self, data: xr.DataArray) -> None:
        if np.any(data.values < self._min_value):
            self._log_or_raise(
                f"Identified values less than the set minimum: {self._min_value}."
            )

    def _ensure_values_below_maximum(self, data: xr.DataArray) -> None:
        if np.any(data.values > self._max_value):
            self._log_or_raise(
                f"Identified values greater than the set maximum: {self._max_value}."
            )

    def _ensure_no_nan_values(self, data: xr.DataArray) -> None:
        if data.isnull().any():
            self._log_or_raise("Identified NaN values!")

    def validate(self, data: xr.DataArray) -> None:
        if self._dim_match_type == EXACT_MATCH_TYPE:
            self._match_dims_exactly(data)
        else:
            self._match_minimum_dims(data)

        if self._coord_match_type == "exact":
            self._match_coords_exactly(data)
        else:
            self._match_minimum_coords(data)

        if not self._nan_values_allowed:
            self._ensure_no_nan_values(data)

        self._ensure_values_above_minimum(data)
        self._ensure_values_below_maximum(data)
