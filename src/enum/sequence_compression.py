from enum import Enum, auto


class SequenceCompressionPoolingType(Enum):
    MEAN = auto()
    MAX = auto()
    BEST = auto()
