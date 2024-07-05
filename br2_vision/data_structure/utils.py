from typing import Callable, Protocol


class ContextProtocol(Protocol):
    _inside_context: bool


def raise_if_outside_context(method: Callable) -> Callable:  # pragma: no cover
    def decorator(self: ContextProtocol, *args, **kwargs):
        if not self._inside_context:
            raise Exception("This method should be called from inside context.")
        return method(self, *args, **kwargs)

    return decorator
