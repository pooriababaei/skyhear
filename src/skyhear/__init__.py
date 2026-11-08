"""
skyhear — Aircraft-Noise Geolocation Tool

Estimates where an audio/video recording was made by analyzing Doppler
signatures of aircraft passing overhead and matching against ADS-B data.
"""

from .adsb import BoundingBox, FlightTrack
from .locate import locate
from .search import CandidateLocation, FlightMatch

__version__ = "0.1.0"
__all__ = [
    "locate",
    "CandidateLocation",
    "FlightMatch",
    "BoundingBox",
    "FlightTrack",
]
