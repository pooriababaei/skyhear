"""
Tests for the ADS-B module.
"""

from __future__ import annotations

import numpy as np
import pytest

from skyhear.adsb import BoundingBox, FlightTrack, SpatialIndex


class TestBoundingBox:
    """Test BoundingBox."""

    def test_creation(self):
        bbox = BoundingBox(24.0, 50.0, -125.0, -66.0)
        assert bbox.min_lat == 24.0
        assert bbox.max_lon == -66.0

    def test_from_center_radius(self):
        bbox = BoundingBox.from_center_radius(40.0, -105.0, 100.0)
        assert bbox.min_lat < 40.0
        assert bbox.max_lat > 40.0
        assert bbox.min_lon < -105.0
        assert bbox.max_lon > -105.0

    def test_contains(self):
        bbox = BoundingBox(24.0, 50.0, -125.0, -66.0)
        assert bbox.contains(40.0, -100.0)
        assert not bbox.contains(60.0, -100.0)

    def test_invalid_bounds(self):
        with pytest.raises(AssertionError):
            BoundingBox(50.0, 24.0, -125.0, -66.0)  # min > max


class TestFlightTrack:
    """Test FlightTrack."""

    def _make_track(self, n=100):
        t = np.arange(n, dtype=float)
        return FlightTrack(
            icao24="abc123",
            callsign="TEST001",
            timestamps=t,
            lat=np.linspace(39.0, 41.0, n),
            lon=np.full(n, -105.0),
            alt_m=np.full(n, 10000.0),
            vx=np.full(n, 100.0),
            vy=np.full(n, 200.0),
            vz=np.zeros(n),
        )

    def test_duration(self):
        track = self._make_track(100)
        assert track.duration_s == pytest.approx(99.0)

    def test_position_at(self):
        track = self._make_track(100)
        pos = track.position_at(50.0)
        assert pos is not None
        lat, lon, alt = pos
        assert lat == pytest.approx(40.0, abs=0.1)

    def test_position_at_out_of_range(self):
        track = self._make_track(100)
        assert track.position_at(-1.0) is None
        assert track.position_at(200.0) is None


class TestSpatialIndex:
    """Test SpatialIndex."""

    def test_query_finds_nearby(self):
        tracks = [
            FlightTrack(
                icao24="abc123",
                callsign="TEST001",
                timestamps=np.arange(100, dtype=float),
                lat=np.full(100, 40.0),
                lon=np.linspace(-105.1, -104.9, 100),
                alt_m=np.full(100, 10000.0),
                vx=np.full(100, 200.0),
                vy=np.zeros(100),
                vz=np.zeros(100),
            )
        ]
        idx = SpatialIndex(tracks, grid_size_deg=0.5)
        results = idx.query(40.0, -105.0, 0.0, 100.0)
        assert len(results) == 1

    def test_query_excludes_far(self):
        tracks = [
            FlightTrack(
                icao24="abc123",
                callsign="TEST001",
                timestamps=np.arange(100, dtype=float),
                lat=np.full(100, 50.0),  # far north
                lon=np.full(100, -105.0),
                alt_m=np.full(100, 10000.0),
                vx=np.full(100, 200.0),
                vy=np.zeros(100),
                vz=np.zeros(100),
            )
        ]
        idx = SpatialIndex(tracks, grid_size_deg=0.5)
        results = idx.query(40.0, -105.0, 0.0, 100.0, max_slant_range_km=10.0)
        assert len(results) == 0


class TestReadsbParser:
    """Test readsb / adsb.lol trace parser."""

    def test_parse_readsb_trace(self):
        from skyhear.adsb import parse_readsb_trace

        raw_data = {
            "icao": "c0ffee",
            "flight": "WJA123",
            "timestamp": 1780887600.0,
            "trace": [
                [0.0, 49.2, -123.1, 10000, 250, 90, 0],
                [10.0, 49.2, -123.05, 10000, 250, 90, 0],
                [20.0, 49.2, -123.0, 10000, 250, 90, 0],
                [30.0, 49.2, -122.95, 10000, 250, 90, 0],
                [40.0, 49.2, -122.9, 10000, 250, 90, 0],
            ],
        }

        track = parse_readsb_trace(raw_data)
        assert track is not None
        assert track.icao24 == "c0ffee"
        assert track.callsign == "WJA123"
        assert len(track.timestamps) == 41  # 0 to 40 inclusive at 1s intervals
        assert track.alt_m[0] == pytest.approx(10000 * 0.3048)

    def test_load_readsb_traces_dir(self, tmp_path):
        import json
        from skyhear.adsb import load_readsb_traces_dir, BoundingBox

        trace_file = tmp_path / "trace_c0ffee.json"
        raw_data = {
            "icao": "c0ffee",
            "flight": "WJA123",
            "timestamp": 1780887600.0,
            "trace": [
                [0.0, 49.2, -123.1, 10000, 250, 90, 0],
                [10.0, 49.2, -123.05, 10000, 250, 90, 0],
                [20.0, 49.2, -123.0, 10000, 250, 90, 0],
                [30.0, 49.2, -122.95, 10000, 250, 90, 0],
                [40.0, 49.2, -122.9, 10000, 250, 90, 0],
            ],
        }
        trace_file.write_text(json.dumps(raw_data))

        bbox = BoundingBox(49.0, 50.0, -124.0, -122.0)
        tracks = load_readsb_traces_dir(tmp_path, bbox=bbox)
        assert len(tracks) == 1
        assert tracks[0].callsign == "WJA123"

    def test_parse_opensky_track(self):
        from skyhear.adsb import parse_opensky_track

        raw_data = {
            "icao24": "c0ffee",
            "callsign": "ACA123",
            "path": [
                [100.0, 49.20, -123.10, 2000.0, 90.0, False],
                [110.0, 49.20, -123.05, 2000.0, 90.0, False],
                [120.0, 49.20, -123.00, 2000.0, 90.0, False],
                [130.0, 49.20, -122.95, 2000.0, 90.0, False],
                [140.0, 49.20, -122.90, 2000.0, 90.0, False],
            ],
        }
        track = parse_opensky_track(raw_data)
        assert track is not None
        assert track.icao24 == "c0ffee"
        assert track.callsign == "ACA123"
        assert len(track.timestamps) == 41  # 100 to 140 inclusive
        assert track.alt_m[0] == pytest.approx(2000.0)


class TestCliRegionParsing:
    """Test CLI region parsing."""

    def test_named_city_region(self):
        from skyhear.cli import _parse_region

        bbox = _parse_region("Prague")
        assert bbox.min_lat < 50.08 < bbox.max_lat
        assert bbox.min_lon < 14.44 < bbox.max_lon

    def test_north_america_full_continent(self):
        from skyhear.cli import _parse_region

        bbox = _parse_region("North America")
        assert bbox.min_lat == 14.5 and bbox.max_lat == 71.5
        assert bbox.min_lon == -168.0 and bbox.max_lon == -52.0


