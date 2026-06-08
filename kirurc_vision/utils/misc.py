def bunched(iterable, n: int = 2):
    """Iterates over the iterable, yielding bunches of n elements each time."""
    it = iter(iterable)
    while True:
        try:
            yield tuple([next(it) for _ in range(n)])
        except StopIteration:
            return
