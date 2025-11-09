#!/usr/bin/env python3

import time


"""
A simple Timer class with context manager support
"""


class Timer:
    def __init__(self, ema=1.0):
        """Timer object for convenient time keeping (courtesy of @mmattamala)

        Args:
            ema (float, optional): weight for exponential moving average ema*meas + (1 - ema)*average. Defaults to 0.0.
        """
        self._ema = ema
        self._tics = {}
        self._keys = []
        self._events = {}

    def __call__(self, key="last"):
        self._keys.append(key)
        return self

    def __str__(self):
        out = ""
        for k, e in self._events.items():
            n = e["count"]
            t = e["time"]
            out += f"{k:<30}: {t:<5.4f}s | calls: {n:<5} | total time: {t*n:<5.4}s\n"
        return out

    def timing(self):
        return time.perf_counter()

    # Context manager interface
    def __enter__(self):
        key = self._keys[-1] if self._keys else "last"
        self.tic(key=key)

    def __exit__(self, exc_type, exc_value, exc_tb):
        key = self._keys[-1]
        self.toc(key=key)

        if key == "last":
            n = self._events[key]["count"]
            t = self._events[key]["time"]
            print(f"time: {t:.4}s | calls: {n:<5} | total time: {t*n:<5.4}s")

    # Normal interface
    def tic(self, key="last"):
        if key not in self._keys:
            self._keys.append(key)
        self._tics[key] = self.timing()

        if key not in self._events:
            self._events[key] = {"count": 0, "time": None}
        self._events[key]["count"] = self._events[key]["count"] + 1

    def toc(self, key="last"):
        toc = self.timing()

        dt = toc - self._tics[key]
        if self._events[key]["time"] is None:
            self._events[key]["time"] = dt
        else:
            self._events[key]["time"] = (
                self._ema * dt + (1.0 - self._ema) * self._events[key]["time"]
            )

        if len(self._keys) > 0:
            self._keys.pop()

        return self._events[key]["time"]

    @property
    def dt(self, key="last"):
        return self._events[key]["time"]

    @property
    def events(self):
        return self._events


if __name__ == "__main__":
    print("Tic-toc interface")
    t = Timer()
    t.tic()
    time.sleep(1)
    t.toc()
    print(t)

    print("\nContext manager interface - case 1")
    with Timer():
        time.sleep(1)

    print("\nContext manager interface - case 2")
    timer = Timer()
    with timer("test0.5"):
        time.sleep(0.5)
    with timer("test1"):
        time.sleep(1)
    print(timer)

    # Average time with context manager
    alpha = 0.5
    print(f"\nContext manager with Exponential Moving average (weight={alpha})")
    timer = Timer(ema=alpha)
    for i in range(10):
        with timer("test_accumulation"):
            print(f"sleep for {i/10} s")
            time.sleep(i / 10)
    print(timer)

    # Nested timers
    print("\nNested timer - case 1")
    with Timer():
        time.sleep(1)
        with Timer():
            time.sleep(2)

    print("\nNested timer - case 2")
    timer = Timer()
    with timer("test1"):
        time.sleep(1)
        with timer("test2"):
            time.sleep(2)
    print(timer)

    print("\nMultiple calls (10)")
    timer = Timer()
    for i in range(10):
        with timer("test1"):
            time.sleep(1)
    print(timer)
