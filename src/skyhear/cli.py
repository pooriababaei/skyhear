"""
skyhear.cli — Command-line interface.

Usage:
    geo-locate --input clip.mp4 --date 2026-09-08 --region bbox:24,-125,50,-66 --top 10 --out results/
    geo-locate --input clip.wav --date 2026-09-08 --region center:40.0,-105.0,radius_km:300
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

logger = logging.getLogger("skyhear")


def _load_env_file(path: Path | None = None) -> None:
    """Load key-value pairs from a .env file into os.environ if not already set."""
    if path is None:
        path = Path.cwd() / ".env"
    if not path.is_file():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_env_file()


def _geocode_name(name: str):
    """
    Geocode any place name (city, region, country, or neighbourhood) into a
    geographic BoundingBox using OpenStreetMap Nominatim.
    """
    from .adsb import BoundingBox

    import urllib.request
    import urllib.parse
    import json
    try:
        q = urllib.parse.quote(name.strip())
        url = f"https://nominatim.openstreetmap.org/search?q={q}&format=json&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "skyhear-acoustic/0.1"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            res = json.loads(resp.read().decode())
            if res and len(res) > 0:
                item = res[0]
                addresstype = item.get("addresstype", "")
                raw_bbox = item.get("boundingbox", [])

                if len(raw_bbox) == 4:
                    min_lat = float(raw_bbox[0])
                    max_lat = float(raw_bbox[1])
                    min_lon = float(raw_bbox[2])
                    max_lon = float(raw_bbox[3])

                    # Large political divisions (countries, states, provinces)
                    if addresstype in ("country", "state", "province", "region"):
                        return BoundingBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)

                    # For cities, municipalities, neighbourhoods, or local areas:
                    # Add a standard ~5 km acoustic boundary buffer (0.045 deg) around municipal limits
                    buffer_deg = 0.045
                    return BoundingBox(
                        min_lat=max(-90.0, min_lat - buffer_deg),
                        max_lat=min(90.0, max_lat + buffer_deg),
                        min_lon=max(-180.0, min_lon - buffer_deg),
                        max_lon=min(180.0, max_lon + buffer_deg),
                    )

                lat = float(item["lat"])
                lon = float(item["lon"])
                return BoundingBox.from_center_radius(lat, lon, 25.0)
    except Exception:
        pass
    return None


def _parse_region(region_str: str, is_fresh: bool = False):
    """
    Parse a region string into a BoundingBox.

    Formats:
        - Plain English city, region, or country name (e.g. "<city-name>", "<country-name>")
        - Explicit bbox: "bbox:min_lat,min_lon,max_lat,max_lon"
        - Explicit center: "center:lat,lon,radius_km:50"
    """
    from .adsb import BoundingBox

    norm = region_str.strip().lower()

    if norm.startswith("bbox:"):
        parts = norm[5:].split(",")
        if len(parts) != 4:
            raise click.BadParameter(
                "bbox format: bbox:min_lat,min_lon,max_lat,max_lon"
            )
        min_lat, min_lon, max_lat, max_lon = map(float, parts)
        return BoundingBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)

    elif norm.startswith("center:"):
        rest = norm[7:]
        parts = rest.split(",")
        if len(parts) < 2:
            raise click.BadParameter(
                "center format: center:lat,lon,radius_km:50"
            )
        lat = float(parts[0])
        lon_and_radius = parts[1:]
        radius_km = 45.0
        lon_str = lon_and_radius[0]
        for part in lon_and_radius:
            if part.startswith("radius_km:"):
                radius_km = float(part.split(":")[1])
            else:
                lon_str = part
        lon = float(lon_str)
        return BoundingBox.from_center_radius(lat, lon, radius_km)

    # Continental North America
    if norm in ("north america", "north-america", "na"):
        click.echo("🌎 Searching across continental North America (14.5°N–71.5°N, 168.0°W–52.0°W)...")
        return BoundingBox(min_lat=14.5, max_lat=71.5, min_lon=-168.0, max_lon=-52.0)

    # Named country, state, city, or area lookup via OpenStreetMap Nominatim
    resolved_bbox = _geocode_name(region_str)
    if resolved_bbox is not None:
        click.echo(
            f"📍 Resolved region '{region_str}' to "
            f"({resolved_bbox.min_lat:.2f}°N–{resolved_bbox.max_lat:.2f}°N, "
            f"{resolved_bbox.min_lon:.2f}°W–{resolved_bbox.max_lon:.2f}°W)"
        )
        return resolved_bbox

    raise click.BadParameter(
        f"Could not resolve region '{region_str}'. "
        "Please provide a specific city/country name, 'North America', "
        "bbox:min_lat,min_lon,max_lat,max_lon, or center:lat,lon,radius_km:N"
    )


@click.command()
@click.argument("audio_file", required=False, type=click.Path(exists=True))
@click.option(
    "--input", "-i",
    "input_opt",
    default=None,
    type=click.Path(exists=True),
    help="Input media file (mp4, mov, wav, ogg, mp3, etc.)",
)
@click.option(
    "--date", "-d",
    default=None,
    help="Recording date in YYYY-MM-DD format (auto-inferred with --fresh)",
)
@click.option(
    "--fresh",
    is_flag=True,
    default=False,
    help="Use live/recent ADS-B data for clips recorded in the past hour",
)
@click.option(
    "--region", "-r",
    default=None,
    help="Search region: city or country name, bbox:..., or center:lat,lon,radius_km:N",
)
@click.option(
    "--top", "-n",
    default=3,
    show_default=True,
    help="Number of top candidates to return",
)
@click.option(
    "--out", "-o",
    "output_dir",
    default="results",
    show_default=True,
    type=click.Path(),
    help="Output directory for results",
)
@click.option(
    "--start-time",
    default=None,
    help="Approximate recording start time in UTC (HH:MM:SS). Helps narrow search.",
)
@click.option(
    "--auto-detect / --manual",
    default=True,
    show_default=True,
    help="Auto-detect ridges (default) or use manual seed picking",
)
@click.option(
    "--min-duration",
    default=10.0,
    show_default=True,
    help="Minimum ridge duration in seconds for Doppler pass detection",
)
@click.option(
    "--adsb-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to local directory with readsb / adsb.lol / tar1090 JSON trace files",
)
@click.option(
    "--opensky-user",
    default=None,
    envvar="OPENSKY_USER",
    help="OpenSky Network username (or set OPENSKY_USER env var)",
)
@click.option(
    "--opensky-pass",
    default=None,
    envvar="OPENSKY_PASS",
    help="OpenSky Network password (or set OPENSKY_PASS env var)",
)
@click.option(
    "-v", "--verbose",
    count=True,
    help="Increase verbosity (-v info, -vv debug)",
)
def main(
    audio_file: str | None,
    input_opt: str | None,
    date: str | None,
    fresh: bool,
    region: str | None,
    top: int,
    output_dir: str,
    start_time: str | None,
    auto_detect: bool,
    min_duration: float,
    adsb_dir: str | None,
    opensky_user: str | None,
    opensky_pass: str | None,
    verbose: int,
):
    """
    Geo-locate a recording by matching aircraft Doppler signatures
    against ADS-B flight data.
    """
    # Configure logging
    log_level = logging.WARNING
    if verbose >= 2:
        log_level = logging.DEBUG
    elif verbose >= 1:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from datetime import datetime, timezone
    from .adsb import BoundingBox, fetch_opensky_tracks, build_flight_tracks
    from .detect import (
        auto_detect_ridges,
        compute_spectrogram,
        plot_spectrogram,
        spectral_whitening,
    )
    import matplotlib.pyplot as plt
    from .ingest import extract_audio, load_audio
    from .report import generate_results_json, print_summary
    from .search import SearchConfig, search_location

    input_file = audio_file or input_opt
    if not input_file:
        raise click.UsageError(
            "Missing input media file. Provide it as an argument or via --input.\n"
            "Example: skyhear clip.ogg --region '<city-or-country>' --fresh"
        )

    # Inspect media file for embedded creation_time
    file_utc_datetime = None
    try:
        import av
        container = av.open(str(input_file))
        ct = container.metadata.get("creation_time")
        if ct:
            file_utc_datetime = datetime.fromisoformat(ct.replace("Z", "+00:00"))
    except Exception:
        pass

    # If the user did not specify --date or --fresh, auto-extract from file
    if not date and not fresh:
        if file_utc_datetime is not None:
            date = file_utc_datetime.strftime("%Y-%m-%d")
            click.echo(f"📅 Auto-detected recording date: {date} (from file)")
        else:
            raise click.UsageError(
                "Could not find embedded recording date in media metadata.\n"
                "Please provide --date YYYY-MM-DD or use --fresh if recorded just now."
            )
    elif fresh:
        if not date:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            click.echo(f"⚡ Fresh mode: auto-selected today's date ({date})")

    if not region:
        raise click.UsageError(
            "Missing required option '--region' / '-r' (e.g. --region '<city-or-country>')."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bbox = _parse_region(region, is_fresh=fresh)

    # Convert user local date & time to true UTC using the target region's timezone
    center_lat = (bbox.min_lat + bbox.max_lat) / 2.0
    center_lon = (bbox.min_lon + bbox.max_lon) / 2.0
    from timezonefinder import TimezoneFinder
    import zoneinfo

    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=center_lat, lng=center_lon) or "UTC"
    region_tz = zoneinfo.ZoneInfo(tz_name)

    # === Step 1: Ingest ===
    ingest_result = extract_audio(input_file, output_dir=output_path / "audio")
    click.echo(f"🎧 Media: {Path(input_file).name} ({ingest_result.duration_seconds:.1f}s)")

    # === Step 2: Detect aircraft tones ===
    audio, sr = load_audio(ingest_result.wav_path)
    Sxx, times, freqs = compute_spectrogram(audio, sr)
    Sxx_w = spectral_whitening(Sxx)

    # Save spectrogram
    spec_path = output_path / "spectrogram.png"

    if auto_detect:
        passes = auto_detect_ridges(Sxx_w, times, freqs, min_ridge_duration_s=min_duration)
    else:
        fig = plot_spectrogram(Sxx_w, times, freqs, title="Whitened Spectrogram")
        fig.savefig(str(spec_path), dpi=150)
        click.echo(f"   Spectrogram saved to {spec_path}")
        sys.exit(0)

    if not passes:
        click.echo("⚠️  No aircraft tones detected.")
        sys.exit(1)

    click.echo(f"🔍 Acoustic Analysis: Detected {len(passes)} Doppler tone(s)")
    fig = plot_spectrogram(Sxx_w, times, freqs, passes=passes, title="Detected Passes")
    fig.savefig(str(spec_path), dpi=150)
    plt.close(fig)

    # === Step 3: Determine recording time ===
    recording_utc_start = None
    if start_time:
        # Interpret user-provided start_time & date in region's local timezone, convert to UTC
        try:
            dt_local = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=region_tz)
        except ValueError:
            dt_local = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M").replace(tzinfo=region_tz)
        dt_utc = dt_local.astimezone(zoneinfo.ZoneInfo("UTC"))
        recording_utc_start = dt_utc.timestamp()
        # Update date to true UTC date for archive retrieval
        date = dt_utc.strftime("%Y-%m-%d")
        click.echo(f"⏱️  Time: {dt_local.strftime('%H:%M:%S')} ({tz_name}) → {dt_utc.strftime('%H:%M:%S UTC')}")
    elif fresh:
        file_mtime = os.path.getmtime(input_file)
        now_ts = datetime.now(timezone.utc).timestamp()
        if now_ts - 7200 <= file_mtime <= now_ts + 60:
            recording_utc_start = file_mtime - ingest_result.duration_seconds
            rec_dt = datetime.fromtimestamp(recording_utc_start, tz=timezone.utc)
            click.echo(f"⏱️  Time: ~{rec_dt.strftime('%H:%M:%S UTC')} (from file)")
        else:
            recording_utc_start = now_ts - ingest_result.duration_seconds
            rec_dt = datetime.fromtimestamp(recording_utc_start, tz=timezone.utc)
            click.echo(f"⏱️  Time: ~{rec_dt.strftime('%H:%M:%S UTC')} (live)")
    else:
        if file_utc_datetime is not None:
            recording_utc_start = file_utc_datetime.timestamp()
            # The container creation_time is already strictly in UTC!
            # Align archive date with true UTC date from file
            date = file_utc_datetime.strftime("%Y-%m-%d")
            local_dt = file_utc_datetime.astimezone(region_tz)
            is_video = any(s.type == "video" for s in container.streams)
            source_label = "video" if is_video else "audio"
            click.echo(
                f"⏱️  Time: {file_utc_datetime.strftime('%H:%M:%S UTC')} "
                f"({local_dt.strftime('%H:%M:%S')} {tz_name}) from {source_label} metadata"
            )

    # === Step 4: Fetch ADS-B data ===
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    is_today = (date == today_utc)

    if adsb_dir is not None:
        from .adsb import load_readsb_traces_dir
        tracks = load_readsb_traces_dir(adsb_dir, bbox=bbox)
    elif fresh or is_today:
        from .adsb import fetch_live_tracks
        tracks = fetch_live_tracks(
            bbox=bbox,
            target_time=recording_utc_start,
            username=opensky_user,
            password=opensky_pass,
        )
    else:
        # 1. Try local cache or GitHub daily archive (adsblol/globe_history)
        from .adsb import download_github_traces
        click.echo(f"✈️  Retrieving historical flight paths for {date}...")
        tracks = download_github_traces(date, bbox=bbox, target_time=recording_utc_start)

        # 2. Fall back to OpenSky Network only if GitHub archive not yet published
        if not tracks:
            try:
                df = fetch_opensky_tracks(
                    date, bbox,
                    username=opensky_user,
                    password=opensky_pass,
                )
                if not df.empty:
                    tracks = build_flight_tracks(df)
            except Exception:
                pass

        # If historical search finds 0 flights, NEVER fall back to live feeds
        if not tracks:
            click.echo(f"⚠️  No flight tracks found for {date} in the target airspace.")
            sys.exit(1)

    if not tracks:
        click.echo("⚠️  No matching flight tracks found.")
        sys.exit(1)

    click.echo(f"✈️  Radar: Loaded {len(tracks)} flight tracks")

    # === Step 5: Search ===
    click.echo("🎯 Geolocating...")
    config = SearchConfig(
        max_clock_offset_s=10.0 if recording_utc_start is not None else 180.0,
    )
    candidates = search_location(
        passes, tracks, bbox, config, recording_utc_start=recording_utc_start
    )

    if not candidates:
        click.echo("⚠️  No candidate locations found.")
        sys.exit(1)

    # === Step 6: Report ===
    from .report import calculate_confidence
    top_conf = calculate_confidence(candidates[0])
    if top_conf < 50:
        click.echo("\n📍 Result (Low Confidence):")
    else:
        click.echo(f"\n📍 Top {min(top, len(candidates))} candidates:")
    summary = print_summary(candidates, top_n=top)

    # Save results
    results_path = generate_results_json(candidates[:top], output_path / "results.json")

    # Save summary
    with open(output_path / "summary.txt", "w") as f:
        f.write(summary)

    # Save overlay plot for top candidate
    try:
        from .report import plot_overlay
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
            output_path=output_path / "overlay.png",
        )
        plt.close(overlay_fig)
        click.echo(f"   Doppler overlay saved to {output_path / 'overlay.png'}")
    except Exception as e:
        logger.warning("Failed to generate overlay plot: %s", e)

    click.echo(f"\n✅ Results saved to {output_path}/")


if __name__ == "__main__":
    main()
