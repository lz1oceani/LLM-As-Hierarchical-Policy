import numpy as np, time, re, signal, math, os, warnings
from numbers import Number
from collections import defaultdict


class TimeoutException(Exception):
    pass


def timeout_call(fn, timeout=1, default=None):
    def handler(signum, frame):
        raise TimeoutException("end of time")

    import signal

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        ret = fn()
    except TimeoutException as e:
        ret = default
        signal.alarm(0)
    except:
        signal.alarm(0)
        ret = default
    signal.alarm(0)
    return ret


class EmptyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def get_current_time():
    import datetime
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time
