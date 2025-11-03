from dataclasses import dataclass
from typing import Literal


@dataclass
class OptionSpec:
    S: float
    K: float
    T: float
    r: float
    q: float
    kind: Literal["call","put"]
    sigma: float
