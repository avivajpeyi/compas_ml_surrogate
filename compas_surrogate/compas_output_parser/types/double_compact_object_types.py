from enum import Enum, auto

# TODO: read directly from COMPAS constants.h


class DCOType(Enum):
    """
    Double Compact Object Types
    """

    BBH = auto()
    CHE_BBH = auto()
    NON_CHE_BBH = auto()
    BNS = auto()
    BHNS = auto()
    all = auto()

    def __repr__(self):
        return f"{self.__class__.name}:{self.name}"
