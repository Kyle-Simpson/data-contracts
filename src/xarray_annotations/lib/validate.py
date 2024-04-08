from typing import Any, Callable

from src.xarray_annotations.lib.schema import Schema


def check_input(schema: Schema, arg_name: str) -> Callable:
    """Validate that the input provided as ``arg_name`` matches the ``schema``."""

    def decorator_result(func: Callable) -> Any:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            schema.resolve_expected_dims(**kwargs)
            schema.resolve_expected_coords(**kwargs)
            schema.validate(kwargs[arg_name])

            return func(*args, **kwargs)

        return wrapper

    return decorator_result


def check_output(schema: Schema) -> Callable:
    """Validate that the result of the decorated function matches the ``schema``."""

    def decorator_result(func: Callable) -> Any:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            schema.resolve_expected_dims(**kwargs)
            schema.resolve_expected_coords(**kwargs)
            schema.validate(result)

            return result

        return wrapper

    return decorator_result
