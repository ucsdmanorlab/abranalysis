from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class AppConfig:
    DEFAULT_TIME_SCALE = 10.0
    DEFAULT_Y_MIN_MICROVOLTS = -5.0
    DEFAULT_Y_MAX_MICROVOLTS = 5.0
    DEFAULT_Y_MIN_NANOVOLTS = -5000.0
    DEFAULT_Y_MAX_NANOVOLTS = 5000.0
    
    SUPPORTED_FILE_TYPES = ["csv", "arf", "asc", "tsv"]
    
    UNITS = ['Microvolts', 'Nanovolts']
    TONE_CLICK_OPTIONS = ["Tone", "Click"]
    LABEL_POSITIONS = ["Left outside", "Right outside", "Right inside", "Off"]
