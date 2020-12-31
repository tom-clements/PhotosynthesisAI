import time
from collections import defaultdict
from functools import wraps

import numpy as np
import hashlib
from typing import List, Union

FUNCTION_TIMINGS = defaultdict(lambda: defaultdict(int))


def time_function(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        time_elapsed = time.time() - start_time
        FUNCTION_TIMINGS[f.__name__]["count"] += 1
        FUNCTION_TIMINGS[f.__name__]["time"] += time_elapsed
        return result

    return decorated


@time_function
def find_array_in_2D_array(array: Union[List, np.ndarray], list_array: np.ndarray) -> List[bool]:
    conditions = [list_array[:, i] == value for i, value in enumerate(array)]
    mask = True
    for c in conditions:
        mask = mask & c
    return mask
    return mask


def hash_text(text: str) -> str:
    return hashlib.sha256(bytes(text, encoding="UTF-8")).hexdigest()[:32]
