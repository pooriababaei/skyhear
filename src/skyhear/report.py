"""
skyhear.report — Generate result reports and visualizations.

Produces JSON results, overlay plots (observed vs predicted curves),
and map links for the top candidate locations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .detect import DetectedPass
from .model import predict_doppler_curve
from .search import CandidateLocation

logger = logging.getLogger(__name__)


def generate_results_json(
    candidates: list[CandidateLocation],
    output_path: str | Path,
) -> Path:
    """
    Write ranked candidates and flight matches to a JSON file.

    Parameters
    ----------
    candidates
        Ranked candidate locations (best first).
    output_path
        Path to write the JSON file.

    Returns
    -------
    Path
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "candidates": [
            {
                "rank": i + 1,
                "lat": c.lat,
                "lon": c.lon,
                "total_score": round(c.total_score, 4),
                "n_unexplained_passes": c.n_unexplained_passes,
                "negative_evidence_penalty": round(c.negative_evidence_penalty, 4),
                "flight_matches": [
                    {
                        "icao24": m.icao24,
                        "callsign": m.callsign,
                        "fitted_f0_hz": round(m.fitted_f0, 2),
                        "fitted_clock_offset_s": round(m.fitted_offset, 2),
                        "rms_residual_hz": round(m.rms_residual_hz, 4),
                    }
                    for m in c.flight_matches
                ],
                "google_maps_link": (
                    f"https://www.google.com/maps?q={c.lat:.6f},{c.lon:.6f}"
                ),
            }
            for i, c in enumerate(candidates)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results written to %s", output_path)
    return output_path


def plot_overlay(
    passes: list[DetectedPass],
    candidate: CandidateLocation,
    flight_tracks: dict[str, dict],
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 6),
) -> plt.Figure:
    """
    Plot observed ridges overlaid with the best-fit predicted Doppler
    curves for the top candidate.

    Parameters
    ----------
    passes
        Detected passes.
    candidate
        The candidate location to visualize.
    flight_tracks
        Dict of icao24 → {timestamps, lat, lon, alt_m} for matched flights.
    output_path
        If provided, save the figure.
    figsize
        Figure size.

    Returns
    -------
    plt.Figure
        The overlay figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot observed ridges
    t_min_obs = float("inf")
    t_max_obs = -float("inf")
    for i, p in enumerate(passes):
        t_min_obs = min(t_min_obs, float(np.min(p.times)))
        t_max_obs = max(t_max_obs, float(np.max(p.times)))
        ax.plot(
            p.times,
            p.freqs,
            "o-",
            markersize=2,
            linewidth=1,
            label=f"Observed Pass {i+1} ({p.duration:.1f}s)",
            alpha=0.7,
        )

    # Plot predicted curves for matched flights
    for match in candidate.flight_matches:
        track = flight_tracks.get(match.icao24)
        if track is None:
            continue

        pred_times, pred_freqs = predict_doppler_curve(
            candidate.lat,
            candidate.lon,
            0.0,  # ground level
            np.array(track["lat"]),
            np.array(track["lon"]),
            np.array(track["alt_m"]),
            np.array(track["timestamps"]),
            f0=match.fitted_f0,
            clock_offset=match.fitted_offset,
        )

        # Align time axis if pred_times is absolute UTC and observed is relative
        if len(pred_times) > 0 and pred_times[0] > 1e6 and t_max_obs < 1e6:
            # find the base offset
            base_t = track["timestamps"][0]
            # align to observation window
            mask = (pred_times >= t_min_obs + base_t - 5.0) & (pred_times <= t_max_obs + base_t + 5.0)
            if np.any(mask):
                pred_times = pred_times[mask] - base_t
                pred_freqs = pred_freqs[mask]

        ax.plot(
            pred_times,
            pred_freqs,
            "--",
            linewidth=2,
            label=f"Predicted: {match.callsign or match.icao24} (f0={match.fitted_f0:.1f} Hz)",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        f"Doppler Overlay — Candidate ({candidate.lat:.5f}, {candidate.lon:.5f}), "
        f"Score: {candidate.total_score:.2f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Overlay plot saved to %s", output_path)

    return fig


def calculate_confidence(candidate: CandidateLocation) -> int:
    """
    Calculate an intuitive confidence score from 0 to 100%.

    Factors:
    - Number of matched flights (multi-aircraft multilateration).
    - Unexplained pass ratio (passes detected vs explained).
    - Average RMS fit residual of matched Doppler curves.
    - Clock offset spread between matched flights.
    """
    n_matches = len(candidate.flight_matches)
    n_unexplained = candidate.n_unexplained_passes
    total_passes = n_matches + n_unexplained

    if n_matches == 0:
        return 0

    # Base confidence based on matches vs total passes
    match_ratio = n_matches / max(1, total_passes)
    
    # Residual factor (lower residual = higher confidence)
    avg_rms = sum(m.rms_residual_hz for m in candidate.flight_matches) / n_matches
    # 0 Hz -> 1.0, 15 Hz -> 0.625, 30 Hz -> 0.25, >=40 Hz -> 0.0
    rms_factor = max(0.0, 1.0 - (avg_rms / 40.0))

    # Multi-plane multilateration bonus:
    # 1 plane = max 55% (corridor ambiguity)
    # 2 planes = up to 80%
    # 3 planes = up to 90%
    # 4+ planes = up to 98%
    if n_matches == 1:
        max_possible = 55.0
    elif n_matches == 2:
        max_possible = 80.0
    elif n_matches == 3:
        max_possible = 90.0
    else:
        max_possible = 98.0

    score = max_possible * (match_ratio ** 1.2) * (0.6 + 0.4 * rms_factor)
    return int(max(0, min(99, round(score))))


def print_summary(candidates: list[CandidateLocation], top_n: int = 5) -> str:
    """
    Generate a human-readable summary of top candidates.
    If confidence is less than 50%, displays only the single top estimate
    with a clear advisory instead of multiple confusing tie points.
    """
    if not candidates:
        return "No candidate locations found."

    top_cand = candidates[0]
    top_confidence = calculate_confidence(top_cand)
    is_low_confidence = top_confidence < 50

    # If confidence is less than 50%, restrict output to 1 candidate
    display_n = 1 if is_low_confidence else top_n
    display_candidates = candidates[:display_n]

    lines = [
        "=" * 70,
        "  AIRCRAFT-NOISE GEOLOCATION RESULTS",
        "=" * 70,
        "",
    ]

    if is_low_confidence:
        lines.append("  ⚠️  LOW CONFIDENCE WARNING: Insufficient distinct crossing flights")
        lines.append("      detected to calculate a tight crosshair. Indoor recordings or")
        lines.append("      single flybys restrict the solution to a broad acoustic corridor.")
        lines.append("")

    for i, c in enumerate(display_candidates):
        conf = calculate_confidence(c)
        lines.append(f"  #{i + 1}  ({c.lat:.6f}, {c.lon:.6f})  Confidence: {conf}/100")
        lines.append(
            f"       https://www.google.com/maps?q={c.lat:.6f},{c.lon:.6f}"
        )

        for m in c.flight_matches:
            lines.append(
                f"       ├─ {m.callsign or m.icao24}: "
                f"f0={m.fitted_f0:.1f} Hz, "
                f"offset={m.fitted_offset:+.1f} s, "
                f"residual={m.rms_residual_hz:.2f} Hz"
            )

        if c.n_unexplained_passes > 0:
            lines.append(f"       └─ {c.n_unexplained_passes} unexplained pass(es)")

        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
