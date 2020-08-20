import numpy as np
from typing import List, Union


def find_array_in_2D_array(array: Union[List, np.ndarray], list_array: np.ndarray) -> bool:
    conditions = [list_array[:, i] == value for i, value in enumerate(array)]
    mask = True
    for c in conditions:
        mask = mask & c
    return mask
