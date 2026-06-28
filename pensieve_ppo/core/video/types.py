"""Common video chunk request typing helpers."""

from abc import ABC
from dataclasses import dataclass, fields, is_dataclass
from typing import Generic, TypeVar, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class VideoChunkRequest(ABC):
    """Base class for video chunk requests."""
    pass


VideoChunkRequestType = TypeVar("VideoChunkRequestType", bound=VideoChunkRequest)


class VideoChunkRequestTyped(ABC, Generic[VideoChunkRequestType]):
    """Interface for objects parameterized by a VideoChunkRequest type."""

    @property
    def request_cls(self) -> type[VideoChunkRequestType]:
        """Concrete request dataclass used by this object."""
        cls = type(self)
        for mro_cls in cls.__mro__:
            for base in getattr(mro_cls, "__orig_bases__", ()):
                origin = get_origin(base)
                if not isinstance(origin, type) or not issubclass(origin, VideoChunkRequestTyped):
                    continue
                args = get_args(base)
                if not args:
                    continue
                request_cls = args[0]
                if isinstance(request_cls, type) and issubclass(request_cls, VideoChunkRequest):
                    return request_cls

        raise TypeError(
            f"{cls.__name__} must declare a concrete VideoChunkRequest type via "
            "VideoChunkRequestTyped[RequestType]"
        )

    @staticmethod
    def request_field_schema(request_cls: type[VideoChunkRequest]) -> tuple[tuple[str, object], ...]:
        """Return the exact dataclass field schema for a request class."""
        if not is_dataclass(request_cls):
            raise TypeError(
                f"Video chunk request class must be a dataclass, got {request_cls!r}"
            )
        type_hints = get_type_hints(request_cls)
        return tuple(
            (field.name, type_hints.get(field.name, field.type))
            for field in fields(request_cls)
        )

    def validate_request_cls_match(
        self,
        other: "VideoChunkRequestTyped",
    ) -> None:
        """Validate that this object and another typed object use equivalent request fields."""
        self_schema = self.request_field_schema(self.request_cls)
        other_schema = self.request_field_schema(other.request_cls)
        if self_schema != other_schema:
            raise TypeError(
                f"{type(self).__name__} request schema {self_schema} does not "
                f"match {type(other).__name__} request schema {other_schema}"
            )
