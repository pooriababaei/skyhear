"""
skyhear.locate — High-level geolocation pipeline.

Exposes the primary programmatic API for locating audio/video recordings.
"""

from __future__ import annotations

import logging
import os
import tempfile
import zoneinfo
from datetime import datetime, timezone
from pathlib import Path

from timezonefinder import TimezoneFinder

from .adsb import (
    BoundingBox,
    FlightTrack,
    build_flight_tracks,
    download_github_traces,
    fetch_live_tracks,
    fetch_opensky_tracks,
    load_readsb_traces_dir,
)
from .detect import (
    auto_detect_ridges,
    compute_spectrogram,
    plot_spectrogram,
    spectral_whitening,
)
from .ingest import extract_audio, load_audio
from .report import generate_results_json, plot_overlay, print_summary
from .search import CandidateLocation, SearchConfig, search_location

logger = logging.getLogger(__name__)


def locate(
    media_file: str | Path,
    region: str,
    date: str | None = None,
    start_time: str | None = None,
    fresh: bool = False,
    top: int = 3,
    output_dir: str | Path | None = None,
    adsb_dir: str | Path | None = None,
) -> list[CandidateLocation]:
    """
    Geo-locate an audio/video recording by matching aircraft Doppler tones against ADS-B flight data.

    Parameters
    ----------
    media_file : str | Path
        Path to the recording file (.mp4, .mov, .wav, .mp3, etc.).
    region : str
        Search region (city, country, 'center:lat,lon,radius_km:N', or 'bbox:min_lat,min_lon,max_lat,max_lon').
    date : str | None, optional
        Recording date in YYYY-MM-DD format. If None and not fresh, auto-detected from media metadata.
    start_time : str | None, optional
        Recording start time in regional local time or UTC (HH:MM or HH:MM:SS).
    fresh : bool, optional
        If True, queries live real-time ADS-B radar feeds for clips recorded in the past hour.
    top : int, optional
        Number of top candidate locations to return (default: 3).
    output_dir : str | Path | None, optional
        If provided, writes results.json, summary.txt, spectrogram.png, and overlay.png to this directory.
    adsb_dir : str | Path | None, optional
        Path to a local directory with readsb JSON traces (offline mode).

    Returns
    -------
    list[CandidateLocation]
        Ranked candidate receiver locations.
    """
    from .cli import _parse_region

    media_path = Path(media_file).resolve()
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    # Inspect media file for embedded creation_time
    file_utc_datetime = None
    try:
        import av
        container = av.open(str(media_path))
        ct = container.metadata.get("creation_time")
        if ct:
            file_utc_datetime = datetime.fromisoformat(ct.replace("Z", "+00:00"))
    except Exception:
        pass

    user_date_provided = bool(date)

    if not date and not fresh:
        if file_utc_datetime is not None:
            date = file_utc_datetime.strftime("%Y-%m-%d")
        else:
            raise ValueError(
                "Could not find embedded recording date in media metadata. "
                "Please provide date='YYYY-MM-DD' or use fresh=True if recorded just now."
            )
    elif fresh:
        if not date:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    bbox = _parse_region(region, is_fresh=fresh)
    if bbox is None:
        raise ValueError(f"Could not resolve search region '{region}'.")

    center_lat = (bbox.min_lat + bbox.max_lat) / 2.0
    center_lon = (bbox.min_lon + bbox.max_lon) / 2.0
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=center_lat, lng=center_lon) or "UTC"
    region_tz = zoneinfo.ZoneInfo(tz_name)

    # Audio extraction: clean temp directory if output_dir is omitted
    temp_dir_ctx = None
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        audio_extract_dir = out_path / "audio"
    else:
        temp_dir_ctx = tempfile.TemporaryDirectory()
        out_path = None
        audio_extract_dir = Path(temp_dir_ctx.name)

    try:
        ingest_result = extract_audio(media_path, output_dir=audio_extract_dir)

        MIN_DURATION_SECONDS = 30.0
        if ingest_result.duration_seconds < MIN_DURATION_SECONDS:
            raise ValueError(
                f"Recording is too short ({ingest_result.duration_seconds:.1f}s). "
                f"Acoustic Doppler geolocation requires at least {MIN_DURATION_SECONDS:.0f} seconds of audio "
                "to capture the aircraft's Closest Point of Approach (CPA) S-curve transition. "
                "Recordings under 30 seconds cannot provide reliable or precise geolocation. "
                "Please provide a recording of 30 to 90 seconds."
            )

        # Detect aircraft tones
        audio, sr = load_audio(ingest_result.wav_path)
        Sxx, times, freqs = compute_spectrogram(audio, sr)
        Sxx_w = spectral_whitening(Sxx)
        passes = auto_detect_ridges(Sxx_w, times, freqs, min_ridge_duration_s=10.0)

        if not passes:
            return []

        # Save spectrogram if output_dir is requested
        if out_path is not None:
            import matplotlib.pyplot as plt
            spec_path = out_path / "spectrogram.png"
            fig = plot_spectrogram(Sxx_w, times, freqs, passes=passes, title="Detected Passes")
            fig.savefig(str(spec_path), dpi=150)
            plt.close(fig)

        # Determine recording time
        recording_utc_start = None
        if start_time:
            try:
                dt_local = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=region_tz)
            except ValueError:
                dt_local = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M").replace(tzinfo=region_tz)
            dt_utc = dt_local.astimezone(zoneinfo.ZoneInfo("UTC"))
            recording_utc_start = dt_utc.timestamp()
            date = dt_utc.strftime("%Y-%m-%d")
        elif fresh:
            file_mtime = os.path.getmtime(media_path)
            now_ts = datetime.now(timezone.utc).timestamp()
            if now_ts - 7200 <= file_mtime <= now_ts + 60:
                recording_utc_start = file_mtime - ingest_result.duration_seconds
            else:
                recording_utc_start = now_ts - ingest_result.duration_seconds
        else:
            if file_utc_datetime is not None:
                if not user_date_provided:
                    recording_utc_start = file_utc_datetime.timestamp()
                    date = file_utc_datetime.strftime("%Y-%m-%d")
                else:
                    file_date = file_utc_datetime.strftime("%Y-%m-%d")
                    if file_date == date:
                        recording_utc_start = file_utc_datetime.timestamp()

        # Fetch ADS-B data
        opensky_user = os.getenv("OPENSKY_USER")
        opensky_pass = os.getenv("OPENSKY_PASS")

        if adsb_dir is not None:
            tracks = load_readsb_traces_dir(adsb_dir, bbox=bbox)
        elif fresh:
            tracks = fetch_live_tracks(
                bbox=bbox,
                target_time=recording_utc_start,
                username=opensky_user,
                password=opensky_pass,
            )
        else:
            tracks = download_github_traces(date, bbox=bbox, target_time=recording_utc_start)
            if not tracks:
                try:
                    df = fetch_opensky_tracks(date, bbox, username=opensky_user, password=opensky_pass)
                    if not df.empty:
                        tracks = build_flight_tracks(df)
                except Exception:
                    pass

        if not tracks:
            return []

        # Search candidates
        config = SearchConfig(
            max_clock_offset_s=10.0 if recording_utc_start is not None else 180.0,
        )
        candidates = search_location(
            passes, tracks, bbox, config, recording_utc_start=recording_utc_start
        )

        if not candidates:
            return []

        # If output_dir is requested, save artifacts
        if out_path is not None:
            generate_results_json(candidates[:top], out_path / "results.json")
            summary = print_summary(candidates, top_n=top)
            with open(out_path / "summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)

            try:
                tracks_dict = {
                    t.icao24: {
                        "timestamps": t.timestamps,
                        "lat": t.lat,
                        "lon": t.lon,
                        "alt_m": t.alt_m,
                    }
                    for t in tracks
                }
                overlay_fig = plot_overlay(
                    passes=passes,
                    candidate=candidates[0],
                    flight_tracks=tracks_dict,
                    output_path=out_path / "overlay.png",
                )
                import matplotlib.pyplot as plt
                plt.close(overlay_fig)
            except Exception as e:
                logger.warning("Failed to generate overlay plot: %s", e)

        return candidates[:top]

    finally:
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()
