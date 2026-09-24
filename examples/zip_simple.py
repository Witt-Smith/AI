"""A small iterator exercise kept separate from the training package."""
from collections.abc import Iterable, Iterator
from typing import Any


def zip_simple(*iterables: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
    """Yield tuples until the shortest iterable is exhausted, like built-in zip."""
    iterators = [iter(iterable) for iterable in iterables]
    while iterators:
        values: list[Any] = []
        for iterator in iterators:
            try:
                values.append(next(iterator))
            except StopIteration:
                return
        yield tuple(values)


if __name__ == "__main__":
    names = ["Alice", "Bob", "Charlie"]
    ages = [25, 30, 35]
    print(list(zip_simple(names, ages)))
