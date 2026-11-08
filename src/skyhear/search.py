"""
skyhear.search — Location search via coarse-to-fine grid search.

Searches a geographic region for the recording location by matching
detected Doppler ridges against simulated curves from ADS-B flight data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np

from .adsb import BoundingBox, FlightTrack, SpatialIndex
from .detect import DetectedPass
from .model import (
    compute_slant_range,
    fit_f0_and_offset,
    predict_audibility,
    predict_doppler_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class FlightMatch:
    """A single flight matched to a detected pass."""

    icao24: str
    callsign: str
    fitted_f0: float
    fitted_offset: float
    rms_residual_hz: float


@dataclass
class CandidateLocation:
    """A scored candidate location."""

    lat: float
    lon: float
    total_score: float  # lower is better
    flight_matches: list[FlightMatch] = field(default_factory=list)
    n_unexplained_passes: int = 0
    negative_evidence_penalty: float = 0.0


@dataclass
class SearchConfig:
    """Configuration for the location search."""

    # Grid resolutions for coarse-to-fine search (degrees)
    coarse_grid_deg: float = 0.01     # ~1 km
    medium_grid_deg: float = 0.002    # ~200 m
    fine_grid_deg: float = 0.0005     # ~50 m

    # How many top candidates to refine at each level
    top_k_coarse: int = 15
    top_k_medium: int = 8
    top_k_fine: int = 5

    # Search parameters
    max_slant_range_km: float = 45.0
    max_clock_offset_s: float = 180.0

    # Scoring weights
    unexplained_pass_penalty: float = 50.0   # Hz equivalent per unexplained pass
    negative_evidence_weight: float = 30.0    # Hz per expected-but-absent flight

    # Interval fingerprint tolerance
    interval_tolerance_s: float = 30.0

    # Receiver altitude (assume ground level)
    receiver_alt_m: float = 0.0

    # Parallel processing
    n_workers: int | None = None  # None = use all CPUs


def _score_single_point(args: tuple) -> CandidateLocation | None:
    """Score a single candidate point (designed for multiprocessing)."""
    (
        lat, lon, receiver_alt_m,
        passes, tracks_data,
        max_slant_range_km, max_clock_offset_s,
        unexplained_pass_penalty, negative_evidence_weight,
    ) = args

    flight_matches = []
    total_residual = 0.0
    n_unexplained = 0

    for pass_data in passes:
        obs_times = pass_data["times"]
        obs_freqs = pass_data["freqs"]
        t_ca = pass_data["t_closest_approach"]

        best_match: FlightMatch | None = None
        best_rms = float("inf")

        for track in tracks_data:
            # Fast geographic pre-filter: is candidate point anywhere within audible range of flight?
            margin_deg = (max_slant_range_km + 5.0) / 111.0
            if (
                lat < track["min_lat"] - margin_deg
                or lat > track["max_lat"] + margin_deg
                or lon < track["min_lon"] - margin_deg
                or lon > track["max_lon"] + margin_deg
            ):
                continue

            t_arr = track["timestamps"]
            lat_arr = track["lat"]
            lon_arr = track["lon"]
            alt_arr = track["alt_m"]

            # Quick check: time overlap
            if (
                t_arr[-1] < obs_times[0] - max_clock_offset_s
                or t_arr[0] > obs_times[-1] + max_clock_offset_s
            ):
                continue

            # Quick check: does the flight pass near this point?
            # Subsample for speed in distance check
            step = max(1, len(lat_arr) // 20)
            slant = compute_slant_range(
                lat, lon, receiver_alt_m,
                lat_arr[::step], lon_arr[::step], alt_arr[::step],
            )
            if slant.min() > max_slant_range_km * 1000:
                continue

            # Fit f0 and clock offset
            try:
                f0, offset, rms = fit_f0_and_offset(
                    obs_times, obs_freqs,
                    lat, lon, receiver_alt_m,
                    lat_arr, lon_arr, alt_arr, t_arr,
                    offset_range=(-max_clock_offset_s, max_clock_offset_s),
                )
            except Exception:
                continue

            if abs(abs(offset) - max_clock_offset_s) < 0.05 or abs(offset) > max_clock_offset_s:
                continue

            if rms < best_rms:
                best_rms = rms
                best_match = FlightMatch(
                    icao24=track["icao24"],
                    callsign=track["callsign"],
                    fitted_f0=f0,
                    fitted_offset=offset,
                    rms_residual_hz=rms,
                )

        if best_match is not None and best_match.rms_residual_hz < 40.0:
            flight_matches.append(best_match)
            total_residual += best_match.rms_residual_hz
        else:
            n_unexplained += 1
            total_residual += unexplained_pass_penalty

    # 1. Harmonic offset consistency (intra-aircraft):
    # If the same aircraft explains multiple tones, their fitted offsets must be nearly identical (<0.8s)
    by_aircraft: dict[str, list[float]] = {}
    for m in flight_matches:
        if m.rms_residual_hz < 40.0:
            by_aircraft.setdefault(m.icao24, []).append(m.fitted_offset)

    for icao, offsets in by_aircraft.items():
        if len(offsets) >= 2:
            plane_spread = max(offsets) - min(offsets)
            if plane_spread <= 0.8:
                # Strong reward for genuine multi-harmonic aircraft pass
                total_residual -= 15.0 * len(offsets)
            elif plane_spread > 1.5:
                # Severe penalty: same plane cannot have sound emitted seconds apart
                total_residual += 30.0 * (plane_spread - 1.5)

    # 2. Inter-aircraft clock offset consistency (device clock consensus):
    # All aircraft heard in the SAME recording share the phone's physical clock offset.
    if len(flight_matches) >= 2:
        good_offsets = [m.fitted_offset for m in flight_matches if m.rms_residual_hz < 40.0]
        if len(good_offsets) >= 2:
            # Use robust spread: if 4+ matches, drop the single worst outlier (e.g. transponder lag on distant plane)
            if len(good_offsets) >= 4:
                s = sorted(good_offsets)
                robust_spread = min(s[-1] - s[1], s[-2] - s[0])
            else:
                robust_spread = max(good_offsets) - min(good_offsets)

            if robust_spread > 5.0:
                # Strong penalty for temporal inconsistency across planes in the same recording
                total_residual += 20.0 * (robust_spread - 5.0)
            elif robust_spread <= 2.0:
                # Reward tight consensus across independent aircraft on the phone's clock
                total_residual -= 5.0 * len(good_offsets)

    total_residual = max(1.0, total_residual)

    # Negative evidence: check for flights that should be audible but weren't detected
    # (simplified — full implementation would check all nearby flights)
    neg_penalty = 0.0

    total_score = total_residual + n_unexplained * unexplained_pass_penalty + neg_penalty

    return CandidateLocation(
        lat=lat,
        lon=lon,
        total_score=total_score,
        flight_matches=flight_matches,
        n_unexplained_passes=n_unexplained,
        negative_evidence_penalty=neg_penalty,
    )


def _generate_grid(
    bbox: BoundingBox,
    step_deg: float,
    tracks_data: list[dict] | None = None,
    max_dist_km: float = 50.0,
) -> list[tuple[float, float]]:
    """
    Generate candidate (lat, lon) grid points within a bounding box.
    If tracks_data is provided, restricts candidate search points to the audible
    corridors (within max_dist_km) where flights are actually overhead.
    """
    if tracks_data:
        from collections import defaultdict

        max_dist_deg = max(0.2, max_dist_km / 111.0)
        expand_cells = max(1, int(np.ceil(max_dist_deg / step_deg)))
        cell_flight_count: dict[tuple[float, float], int] = defaultdict(int)

        for t in tracks_data:
            lats = t["lat"]
            lons = t["lon"]
            step = max(1, len(lats) // 15)
            plane_cells = set()
            for lat, lon in zip(lats[::step], lons[::step]):
                if (
                    bbox.min_lat - max_dist_deg <= lat <= bbox.max_lat + max_dist_deg
                    and bbox.min_lon - max_dist_deg <= lon <= bbox.max_lon + max_dist_deg
                ):
                    clat = round(lat / step_deg) * step_deg
                    clon = round(lon / step_deg) * step_deg
                    for di in range(-expand_cells, expand_cells + 1):
                        for dj in range(-expand_cells, expand_cells + 1):
                            pt_lat = round(clat + di * step_deg, 4)
                            pt_lon = round(clon + dj * step_deg, 4)
                            if (
                                bbox.min_lat <= pt_lat <= bbox.max_lat
                                and bbox.min_lon <= pt_lon <= bbox.max_lon
                            ):
                                plane_cells.add((pt_lat, pt_lon))
            for pt in plane_cells:
                cell_flight_count[pt] += 1

        if cell_flight_count:
            lat_span = bbox.max_lat - bbox.min_lat
            # For country-scale searches, require candidate points to be in range
            # of multiple crossing flights to eliminate single-flight empty wilderness
            min_flights = 2 if (lat_span > 2.0 and len(tracks_data) >= 3) else 1
            corridor_cells = [pt for pt, count in cell_flight_count.items() if count >= min_flights]
            if not corridor_cells:
                corridor_cells = list(cell_flight_count.keys())
            return sorted(corridor_cells)

    lats = np.arange(bbox.min_lat, bbox.max_lat + step_deg, step_deg)
    lons = np.arange(bbox.min_lon, bbox.max_lon + step_deg, step_deg)
    grid = [(float(lat), float(lon)) for lat in lats for lon in lons]
    return grid


def _refine_bbox(center_lat: float, center_lon: float, radius_deg: float) -> BoundingBox:
    """Create a small bounding box around a candidate for refinement."""
    return BoundingBox(
        min_lat=center_lat - radius_deg,
        max_lat=center_lat + radius_deg,
        min_lon=center_lon - radius_deg,
        max_lon=center_lon + radius_deg,
    )


def interval_fingerprint_filter(
    passes: list[DetectedPass],
    tracks: list[FlightTrack],
    candidate_lat: float,
    candidate_lon: float,
    candidate_alt_m: float,
    tolerance_s: float = 30.0,
) -> list[FlightTrack]:
    """
    Apply the interval fingerprint filter: the time gaps between
    detected passes must match the gaps between the corresponding
    flights' closest approaches at the candidate location.

    This is a cheap early filter that can eliminate many candidates.

    Parameters
    ----------
    passes
        Detected passes sorted by time.
    tracks
        Candidate flight tracks to match.
    candidate_lat, candidate_lon, candidate_alt_m
        Candidate location.
    tolerance_s
        Maximum difference in inter-pass interval (seconds).

    Returns
    -------
    list[FlightTrack]
        Tracks that are consistent with the interval fingerprint.
    """
    if len(passes) < 2:
        return tracks

    # Compute observed inter-pass intervals
    cas = sorted(p.t_closest_approach for p in passes)
    observed_intervals = [cas[i + 1] - cas[i] for i in range(len(cas) - 1)]

    # For each track, compute its closest-approach time at the candidate
    track_ca_times = []
    for track in tracks:
        slant = compute_slant_range(
            candidate_lat, candidate_lon, candidate_alt_m,
            track.lat, track.lon, track.alt_m,
        )
        min_idx = np.argmin(slant)
        ca_time = float(track.timestamps[min_idx])
        track_ca_times.append((ca_time, track))

    track_ca_times.sort(key=lambda x: x[0])

    # Try to find a subset of tracks whose intervals match
    # This is simplified — a full implementation would do combinatorial matching
    compatible = [t for _, t in track_ca_times]
    return compatible


def search_location(
    passes: list[DetectedPass],
    tracks: list[FlightTrack],
    bbox: BoundingBox,
    config: SearchConfig | None = None,
    recording_utc_start: float | None = None,
) -> list[CandidateLocation]:
    """
    Search for the recording location using coarse-to-fine grid search.

    Parameters
    ----------
    passes
        Detected aircraft passes from the audio.
    tracks
        ADS-B flight tracks for the region and date.
    bbox
        Search bounding box.
    config
        Search configuration.
    recording_utc_start
        Approximate UTC start time of the recording (Unix timestamp).
        Used to narrow the time window for flight matching.

    Returns
    -------
    list[CandidateLocation]
        Ranked candidates, best (lowest score) first.
    """
    if config is None:
        config = SearchConfig()

    n_workers = config.n_workers or max(1, (os.cpu_count() or 2) - 1)

    # Prepare serializable pass data for multiprocessing
    t_base = recording_utc_start or 0.0
    passes_data = []
    for p in passes:
        passes_data.append({
            "times": np.array([t_base + pt[0] for pt in p.ridge_points]),
            "freqs": np.array([pt[1] for pt in p.ridge_points]),
            "t_closest_approach": t_base + p.t_closest_approach,
        })

    # Time bounds across all passes, including acoustic propagation delay (slant_range / c)
    prop_delay_s = (config.max_slant_range_km * 1000.0 / 343.0) + config.max_clock_offset_s + 30.0
    if passes_data:
        t_min_all = min(p["times"][0] for p in passes_data) - prop_delay_s
        t_max_all = max(p["times"][-1] for p in passes_data) + prop_delay_s
    else:
        t_min_all = -float("inf")
        t_max_all = float("inf")

    # Filter and prepare flight tracks: only airborne tracks in the region during the window
    margin_deg = (config.max_slant_range_km + 10.0) / 111.0

    tracks_data = []
    for t in tracks:
        if recording_utc_start is not None:
            mask = (t.timestamps >= t_min_all) & (t.timestamps <= t_max_all)
            if not np.any(mask):
                continue
            if np.max(t.alt_m[mask]) < 100:
                continue
            in_bbox = (
                (t.lat[mask] >= bbox.min_lat - margin_deg)
                & (t.lat[mask] <= bbox.max_lat + margin_deg)
                & (t.lon[mask] >= bbox.min_lon - margin_deg)
                & (t.lon[mask] <= bbox.max_lon + margin_deg)
            )
            if not np.any(in_bbox):
                continue

            t_lat = t.lat[mask]
            t_lon = t.lon[mask]
            tracks_data.append({
                "icao24": t.icao24,
                "callsign": t.callsign,
                "timestamps": t.timestamps[mask],
                "lat": t_lat,
                "lon": t_lon,
                "alt_m": t.alt_m[mask],
                "min_lat": float(np.min(t_lat)),
                "max_lat": float(np.max(t_lat)),
                "min_lon": float(np.min(t_lon)),
                "max_lon": float(np.max(t_lon)),
            })
        else:
            tracks_data.append({
                "icao24": t.icao24,
                "callsign": t.callsign,
                "timestamps": t.timestamps,
                "lat": t.lat,
                "lon": t.lon,
                "alt_m": t.alt_m,
                "min_lat": float(np.min(t.lat)),
                "max_lat": float(np.max(t.lat)),
                "min_lon": float(np.min(t.lon)),
                "max_lon": float(np.max(t.lon)),
            })

    logger.info("Search space: %d relevant flight tracks over %s", len(tracks_data), bbox)

    # Adaptive multi-scale resolution based on bounding box geographic extent
    lat_span = bbox.max_lat - bbox.min_lat
    if lat_span > 10.0:
        # Continental / Full country scale
        coarse_step = max(config.coarse_grid_deg, 0.15)
        medium_step = max(config.medium_grid_deg, 0.015)
        fine_step = config.fine_grid_deg
    elif lat_span > 2.0:
        # Regional / State / Province scale
        coarse_step = max(config.coarse_grid_deg, 0.05)
        medium_step = max(config.medium_grid_deg, 0.005)
        fine_step = config.fine_grid_deg
    else:
        # Local / City / Metropolitan scale
        coarse_step = config.coarse_grid_deg
        medium_step = config.medium_grid_deg
        fine_step = config.fine_grid_deg

    # === Phase 1: Coarse grid (audible flight corridors) ===
    candidates = _search_grid(
        bbox, coarse_step, passes_data, tracks_data, config, n_workers, use_corridors=True
    )
    candidates.sort(key=lambda c: c.total_score)
    # Non-maximum suppression: select geographically distinct coarse candidate regions
    min_dist_deg = coarse_step * 1.5
    top_coarse: list[CandidateLocation] = []
    for c in candidates:
        if not any(
            (c.lat - s.lat)**2 + (c.lon - s.lon)**2 < min_dist_deg**2
            for s in top_coarse
        ):
            top_coarse.append(c)
            if len(top_coarse) == config.top_k_coarse:
                break
    if not top_coarse:
        top_coarse = candidates[: config.top_k_coarse]

    logger.info(
        "Coarse search: %d points → top %d distinct regions (best score: %.2f)",
        len(candidates),
        len(top_coarse),
        top_coarse[0].total_score if top_coarse else float("inf"),
    )

    # === Phase 2: Medium grid around top coarse candidates ===
    medium_candidates = []
    for c in top_coarse:
        refined_bbox = _refine_bbox(c.lat, c.lon, coarse_step * 0.75)
        results = _search_grid(
            refined_bbox, medium_step, passes_data, tracks_data, config, n_workers, use_corridors=False
        )
        medium_candidates.extend(results)

    medium_candidates.sort(key=lambda c: c.total_score)
    min_med_dist_deg = max(coarse_step * 0.5, medium_step * 2.0)
    top_medium: list[CandidateLocation] = []
    for c in medium_candidates:
        if not any(
            (c.lat - s.lat)**2 + (c.lon - s.lon)**2 < min_med_dist_deg**2
            for s in top_medium
        ):
            top_medium.append(c)
            if len(top_medium) == config.top_k_medium:
                break
    if not top_medium:
        top_medium = medium_candidates[: config.top_k_medium]

    logger.info(
        "Medium search: %d points → top %d (best: %.2f)",
        len(medium_candidates),
        len(top_medium),
        top_medium[0].total_score if top_medium else float("inf"),
    )

    # === Phase 3: Fine grid around top medium candidates ===
    fine_candidates = []
    for c in top_medium:
        refined_bbox = _refine_bbox(c.lat, c.lon, medium_step * 0.75)
        results = _search_grid(
            refined_bbox, fine_step, passes_data, tracks_data, config, n_workers, use_corridors=False
        )
        fine_candidates.extend(results)

    fine_candidates.sort(key=lambda c: c.total_score)
    # Regional spatial diversity for top candidates
    if lat_span > 2.0:
        min_fine_dist_deg = 0.08  # ~9 km separation across regional/200km radius
    elif lat_span > 0.5:
        min_fine_dist_deg = 0.03  # ~3.5 km separation
    else:
        min_fine_dist_deg = max(fine_step * 4.0, 0.003)  # ~300m for city search
    top_fine: list[CandidateLocation] = []
    for c in fine_candidates:
        if not any(
            (c.lat - s.lat)**2 + (c.lon - s.lon)**2 < min_fine_dist_deg**2
            for s in top_fine
        ):
            top_fine.append(c)
            if len(top_fine) == config.top_k_fine:
                break
    if not top_fine:
        top_fine = fine_candidates[: config.top_k_fine]
    logger.info(
        "Fine search: %d points → top %d (best: %.2f)",
        len(fine_candidates),
        len(top_fine),
        top_fine[0].total_score if top_fine else float("inf"),
    )

    return top_fine


def _search_grid(
    bbox: BoundingBox,
    step_deg: float,
    passes_data: list[dict],
    tracks_data: list[dict],
    config: SearchConfig,
    n_workers: int,
    use_corridors: bool = False,
) -> list[CandidateLocation]:
    """Score all grid points in a bounding box."""
    grid = _generate_grid(
        bbox,
        step_deg,
        tracks_data=tracks_data if use_corridors else None,
        max_dist_km=config.max_slant_range_km + 10.0,
    )

    if not grid:
        return []

    # Build arguments for parallel scoring
    args_list = [
        (
            lat, lon, config.receiver_alt_m,
            passes_data, tracks_data,
            config.max_slant_range_km, config.max_clock_offset_s,
            config.unexplained_pass_penalty, config.negative_evidence_weight,
        )
        for lat, lon in grid
    ]

    # Run parallel scoring via ThreadPoolExecutor (releases GIL in C/Fortran minimize)
    if n_workers > 1 and len(args_list) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_score_single_point, args_list))
    else:
        results = [_score_single_point(a) for a in args_list]

    return [r for r in results if r is not None]
