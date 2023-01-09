from typing import Dict, List, Tuple, Union

import numpy as np
from bilby.core.prior import PriorDict, Uniform

STAR_FORMATION_PRIOR = PriorDict(
    dict(
        aSF=Uniform(name="aSF", minimum=0.0, maximum=1.0),
        bSF=Uniform(name="bSF", minimum=0.0, maximum=1.0),
        cSF=Uniform(name="cSF", minimum=0.0, maximum=1.0),
        dSF=Uniform(name="dSF", minimum=0.5, maximum=6, latex_label="$dSF$"),
        alpha=Uniform(name="alpha", minimum=0.0, maximum=1.0),
        beta=Uniform(name="beta", minimum=0.0, maximum=1.0),
    )
)


def draw_star_formation_samples(n=1000) -> Dict[str, np.ndarray]:
    samples = STAR_FORMATION_PRIOR.sample(n)
    return samples
