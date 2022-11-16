from enum import Enum, auto


class StellarType(Enum):
    """Obtained from COMPAS.constants.StellarType"""

    MS_LTE_07 = auto()
    MS_GT_07 = auto()
    HERTZSPRUNG_GAP = auto()
    FIRST_GIANT_BRANCH = auto()
    CORE_HELIUM_BURNING = auto()
    EARLY_ASYMPTOTIC_GIANT_BRANCH = auto()
    THERMALLY_PULSING_ASYMPTOTIC_GIANT_BRANCH = auto()
    NAKED_HELIUM_STAR_MS = auto()
    NAKED_HELIUM_STAR_HERTZSPRUNG_GAP = auto()
    NAKED_HELIUM_STAR_GIANT_BRANCH = auto()
    HELIUM_WHITE_DWARF = auto()
    CARBON_OXYGEN_WHITE_DWARF = auto()
    OXYGEN_NEON_WHITE_DWARF = auto()
    NEUTRON_STAR = auto()
    BLACK_HOLE = auto()
    MASSLESS_REMNANT = auto()
    CHEMICALLY_HOMOGENEOUS = auto()
    STAR = auto()
    BINARY_STAR = auto()
    NONE = auto()

    def __repr__(self):
        return f"{type(self).__name__}:{self.name}"
