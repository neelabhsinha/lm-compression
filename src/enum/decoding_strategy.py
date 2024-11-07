from enum import Enum, auto


class DecodingStrategy(Enum):
    TOP_K = auto()
    TOP_P = auto()
    GREEDY = auto()
    RANDOM = auto()
