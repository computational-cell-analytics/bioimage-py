"""Protocols and validation for bounded associative reductions."""
from __future__ import annotations

import numbers
from typing import Generic, Optional, Protocol, TypeVar


ValueT = TypeVar("ValueT")
AccumulatorT = TypeVar("AccumulatorT")
ResultT = TypeVar("ResultT")

DEFAULT_REDUCTION_BATCH_SIZE = 1_000


class Reducer(Protocol, Generic[ValueT, AccumulatorT, ResultT]):
    """Combine logical results through a bounded accumulator.

    The runner calls ``initial`` once for each batch and once for the final fold. It processes
    values and batch accumulators in logical order. Implementations can mutate an accumulator,
    but each method must return the accumulator that subsequent calls use. Methods must not mutate
    the reducer object because local workers can call them concurrently. Distributed reducers and
    accumulators must support ``cloudpickle`` serialization.
    """

    def initial(self) -> AccumulatorT:
        """Return a new identity accumulator."""
        ...

    def update(self, accumulator: AccumulatorT, value: ValueT) -> AccumulatorT:
        """Add one logical result to an accumulator."""
        ...

    def merge(self, left: AccumulatorT, right: AccumulatorT) -> AccumulatorT:
        """Merge two accumulators and preserve their logical order."""
        ...

    def finalize(self, accumulator: AccumulatorT) -> ResultT:
        """Convert the complete accumulator to the public result."""
        ...


def _normalize_reduction_batch_size(value: Optional[int]) -> int:
    """Validate a reduction batch size and apply the default."""
    if value is None:
        return DEFAULT_REDUCTION_BATCH_SIZE
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("reduction_batch_size must be an integer.")
    value = int(value)
    if value <= 0:
        raise ValueError("reduction_batch_size must be positive.")
    return value
