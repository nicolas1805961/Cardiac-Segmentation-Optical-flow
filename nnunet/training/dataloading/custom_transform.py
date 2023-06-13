import monai.transforms as T
from monai.transforms import TraceableTransform
from monai.transforms import Compose
import numpy as np
import matplotlib.pyplot as plt
from monai.config import KeysCollection
from typing import Optional, Any, Mapping, Hashable
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.config.type_definitions import NdarrayOrTensor

from monai.transforms import (
    Transform,
    Randomizable,
    Compose,
)

class RandInvert(Randomizable, Transform):
    def __init__(self, prob: float) -> None:
        self.prob = np.clip(prob, 0.0, 1.0)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return

class RandInvertd(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, prob: float) -> None:
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob)
        self.transform = RandInvert(prob)
    
    def set_random_state(self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None):
        super().set_random_state(seed, state)
        self.transform.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]):
        d = dict(data)
        if self.R.random() < self.prob:
            for key in self.keys:
                d[key] = -d[key]
        return d