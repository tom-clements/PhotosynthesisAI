from dataclasses import dataclass

@dataclass
class Token:
    richness: int
    value: int
    owner: int = None
