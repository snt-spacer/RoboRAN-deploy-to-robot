import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg

@dataclass
class HippopedeTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    a: float = 1.0
    b: float = 1.0

class HippopedeTrajectory(BaseTrajectory, Registerable):
    _cfg: HippopedeTrajectoryCfg
    def __init__(self, cfg: HippopedeTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        theta = np.linspace(0, 2 * np.pi, num=1000)
        r = 4*self._cfg.b*(self._cfg.a - self._cfg.b * np.sin(theta)**2)
        
