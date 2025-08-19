# Consider using dataclasses or NamedTuple for better structure
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class InputSettings:
    time_scale: float
    level: bool
    units: str
    calibration_levels: Dict[tuple, float]
    multiply_y_factor: float

@dataclass 
class OutputSettings:
    return_units: str
    smooth_on: bool
    all_peaks: bool
    serif_font: bool
    show_peaks: bool
    show_legend: bool
    plot_time_warped: bool
    auto_y: bool
    y_min: float
    y_max: float
    vert_space: float
