"""
skyhear.adsb — ADS-B flight data acquisition and management.

Downloads, caches, and indexes historical ADS-B flight tracks
for matching against detected aircraft passes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".skyhear" / "adsb_cache"


def _safe_urlopen(req: urllib.request.Request | str, timeout: float = 10):
    """Open URL using certifi CA bundle if available, falling back to unverified context on SSL error."""
    import ssl
    import urllib.request

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()

    try:
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)
    except Exception as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc) or "certificate verify failed" in str(exc):
            unverified_ctx = ssl._create_unverified_context()
            return urllib.request.urlopen(req, timeout=timeout, context=unverified_ctx)
        raise


@dataclass
class BoundingBox:
    """Geographic bounding box."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def __post_init__(self):
        assert -90 <= self.min_lat <= self.max_lat <= 90
        assert -180 <= self.min_lon <= self.max_lon <= 180

    @classmethod
    def from_center_radius(cls, lat: float, lon: float, radius_km: float) -> BoundingBox:
        """Create a bounding box from center + radius (approximate)."""
        # Rough approximation: 1 degree lat ≈ 111 km
        dlat = radius_km / 111.0
        # Longitude degrees vary with latitude
        dlon = radius_km / (111.0 * np.cos(np.radians(lat)))
        return cls(
            min_lat=lat - dlat,
            max_lat=lat + dlat,
            min_lon=lon - dlon,
            max_lon=lon + dlon,
        )

    def contains(self, lat: float, lon: float) -> bool:
        return (self.min_lat <= lat <= self.max_lat) and (self.min_lon <= lon <= self.max_lon)


@dataclass
class FlightTrack:
    """A single flight's interpolated track."""

    icao24: str
    callsign: str
    # Uniform 1-second timeline (UTC timestamps)
    timestamps: np.ndarray  # float64, Unix epoch seconds
    lat: np.ndarray
    lon: np.ndarray
    alt_m: np.ndarray  # geometric altitude in meters
    vx: np.ndarray  # velocity east (m/s)
    vy: np.ndarray  # velocity north (m/s)
    vz: np.ndarray  # vertical rate (m/s)

    @property
    def duration_s(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    def position_at(self, t: float) -> tuple[float, float, float] | None:
        """Interpolate position at Unix time t. Returns (lat, lon, alt_m) or None if out of range."""
        if t < self.timestamps[0] or t > self.timestamps[-1]:
            return None
        lat = float(np.interp(t, self.timestamps, self.lat))
        lon = float(np.interp(t, self.timestamps, self.lon))
        alt = float(np.interp(t, self.timestamps, self.alt_m))
        return lat, lon, alt

    def velocity_at(self, t: float) -> tuple[float, float, float] | None:
        """Interpolate velocity at Unix time t. Returns (vx, vy, vz) or None."""
        if t < self.timestamps[0] or t > self.timestamps[-1]:
            return None
        vx = float(np.interp(t, self.timestamps, self.vx))
        vy = float(np.interp(t, self.timestamps, self.vy))
        vz = float(np.interp(t, self.timestamps, self.vz))
        return vx, vy, vz


def _cache_key(date: str, bbox: BoundingBox) -> str:
    """Generate a cache key for a (date, bbox) pair."""
    return (
        f"{date}_"
        f"{bbox.min_lat:.2f}_{bbox.max_lat:.2f}_"
        f"{bbox.min_lon:.2f}_{bbox.max_lon:.2f}"
    )


def _cache_path(date: str, bbox: BoundingBox, cache_dir: Path) -> Path:
    """Return the Parquet cache file path."""
    return cache_dir / f"{_cache_key(date, bbox)}.parquet"


def fetch_opensky_tracks(
    date: str,
    bbox: BoundingBox,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    username: str | None = None,
    password: str | None = None,
) -> pd.DataFrame:
    """
    Fetch ADS-B state vectors from the OpenSky Network REST API for a given
    date and bounding box. Results are cached as Parquet.

    Parameters
    ----------
    date
        Date in "YYYY-MM-DD" format.
    bbox
        Geographic bounding box.
    cache_dir
        Directory for caching Parquet files.
    username, password
        OpenSky credentials (optional, increases rate limits).

    Returns
    -------
    pd.DataFrame
        Columns: icao24, callsign, time (unix), lat, lon, baro_altitude,
        geo_altitude, velocity, heading, vertical_rate, on_ground.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _cache_path(date, bbox, cache_dir)

    if cached.exists():
        logger.info("Loading cached ADS-B data: %s", cached)
        return pd.read_parquet(cached)

    logger.info("Fetching ADS-B data from OpenSky for %s, bbox=%s", date, bbox)

    # Parse date to unix timestamps (full day)
    day_start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    begin = int(day_start.timestamp())
    end = begin + 86400

    # OpenSky REST API: /api/states/all with time parameter
    # For historical data we query in 1-hour chunks
    all_records = []
    base_url = "https://opensky-network.org/api/states/all"

    auth = (username, password) if username and password else None

    for hour_offset in range(24):
        t = begin + hour_offset * 3600
        params = {
            "time": t,
            "lamin": bbox.min_lat,
            "lamax": bbox.max_lat,
            "lomin": bbox.min_lon,
            "lomax": bbox.max_lon,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(base_url, params=params, auth=auth)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("OpenSky rate limited (429), skipping...")
                break
            elif exc.response.status_code in (401, 403):
                logger.debug("OpenSky returned HTTP %d (no credentials or forbidden); skipping OpenSky.", exc.response.status_code)
                break
            break
        except httpx.RequestError as exc:
            logger.warning("Request error at hour %d: %s", hour_offset, exc)
            continue

        states = data.get("states", [])
        if states is None:
            continue

        for sv in states:
            # OpenSky state vector format:
            # [icao24, callsign, origin_country, time_position, last_contact,
            #  longitude, latitude, baro_altitude, on_ground, velocity,
            #  true_track, vertical_rate, sensors, geo_altitude,
            #  squawk, spi, position_source]
            if sv[5] is None or sv[6] is None:
                continue  # No position data

            all_records.append({
                "icao24": sv[0],
                "callsign": (sv[1] or "").strip(),
                "time": sv[3] or t,
                "lon": sv[5],
                "lat": sv[6],
                "baro_altitude": sv[7],
                "on_ground": sv[8],
                "velocity": sv[9],
                "heading": sv[10],
                "vertical_rate": sv[11],
                "geo_altitude": sv[13],
            })

        logger.info(
            "Hour %02d: %d state vectors (%d total)",
            hour_offset,
            len(states) if states else 0,
            len(all_records),
        )

        # Respect rate limits
        if not auth:
            time.sleep(5)  # 5s delay for anonymous queries

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning("No ADS-B data found for %s in the given bbox", date)
        return df

    # Sort by time
    df.sort_values(["icao24", "time"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Cache as Parquet
    df.to_parquet(cached, index=False)
    logger.info("Cached %d records → %s", len(df), cached)

    return df


def build_flight_tracks(
    df: pd.DataFrame,
    min_duration_s: float = 30.0,
    max_gap_s: float = 120.0,
) -> list[FlightTrack]:
    """
    Convert raw ADS-B state vectors into per-flight interpolated tracks.

    Parameters
    ----------
    df
        DataFrame from fetch_opensky_tracks.
    min_duration_s
        Discard tracks shorter than this.
    max_gap_s
        Split tracks at gaps larger than this.

    Returns
    -------
    list[FlightTrack]
        One FlightTrack per continuous flight segment.
    """
    if df.empty:
        return []

    tracks = []

    for icao24, group in df.groupby("icao24"):
        group = group.sort_values("time").dropna(subset=["lat", "lon"])

        if group["on_ground"].all():
            continue  # Skip ground-only tracks

        # Filter out ground segments
        group = group[~group["on_ground"]].copy()

        if len(group) < 2:
            continue

        callsign = group["callsign"].mode().iloc[0] if len(group["callsign"].mode()) > 0 else ""

        # Split on large time gaps
        time_vals = group["time"].values.astype(float)
        gaps = np.diff(time_vals)
        split_indices = np.where(gaps > max_gap_s)[0] + 1
        segments = np.split(np.arange(len(group)), split_indices)

        for seg_indices in segments:
            if len(seg_indices) < 2:
                continue

            seg = group.iloc[seg_indices]
            t = seg["time"].values.astype(float)

            if t[-1] - t[0] < min_duration_s:
                continue

            # Interpolate to 1-second timeline
            t_interp = np.arange(t[0], t[-1] + 1, 1.0)

            lat_interp = np.interp(t_interp, t, seg["lat"].values)
            lon_interp = np.interp(t_interp, t, seg["lon"].values)

            # Altitude: prefer geometric, fall back to barometric
            alt = seg["geo_altitude"].fillna(seg["baro_altitude"]).fillna(10000).values
            alt_interp = np.interp(t_interp, t, alt)

            # Velocity components from speed + heading
            velocity = seg["velocity"].fillna(0).values
            heading_rad = np.radians(seg["heading"].fillna(0).values)
            vx_raw = velocity * np.sin(heading_rad)  # east component
            vy_raw = velocity * np.cos(heading_rad)  # north component
            vz_raw = seg["vertical_rate"].fillna(0).values

            vx_interp = np.interp(t_interp, t, vx_raw)
            vy_interp = np.interp(t_interp, t, vy_raw)
            vz_interp = np.interp(t_interp, t, vz_raw)

            track = FlightTrack(
                icao24=str(icao24),
                callsign=callsign,
                timestamps=t_interp,
                lat=lat_interp,
                lon=lon_interp,
                alt_m=alt_interp,
                vx=vx_interp,
                vy=vy_interp,
                vz=vz_interp,
            )
            tracks.append(track)

    logger.info("Built %d flight tracks from %d unique aircraft", len(tracks), df["icao24"].nunique())
    return tracks


class SpatialIndex:
    """
    Simple spatial index for quickly finding flights near a point within a time window.

    Builds a grid-based index for fast spatial queries.
    """

    def __init__(self, tracks: list[FlightTrack], grid_size_deg: float = 0.5):
        """
        Build a spatial index over a list of flight tracks.

        Parameters
        ----------
        tracks
            Flight tracks to index.
        grid_size_deg
            Grid cell size in degrees for the spatial index.
        """
        self.tracks = tracks
        self.grid_size = grid_size_deg
        self._index: dict[tuple[int, int], list[int]] = {}

        for i, track in enumerate(tracks):
            # Index each track by the grid cells it passes through
            cells = set()
            for lat, lon in zip(track.lat[::10], track.lon[::10]):  # subsample
                cell = (int(lat / grid_size_deg), int(lon / grid_size_deg))
                cells.add(cell)
            for cell in cells:
                self._index.setdefault(cell, []).append(i)

        logger.info("Spatial index: %d cells, %d tracks", len(self._index), len(tracks))

    def query(
        self,
        lat: float,
        lon: float,
        time_start: float,
        time_end: float,
        max_slant_range_km: float = 60.0,
    ) -> list[FlightTrack]:
        """
        Find flights passing near a point within a time window.

        Parameters
        ----------
        lat, lon
            Ground point.
        time_start, time_end
            Unix time window.
        max_slant_range_km
            Maximum slant range to consider (km).

        Returns
        -------
        list[FlightTrack]
            Matching flight tracks.
        """
        # Find candidate cells (include neighbors for border effects)
        center_cell = (int(lat / self.grid_size), int(lon / self.grid_size))
        candidate_indices: set[int] = set()

        # Search a radius of cells
        search_cells = max(1, int(max_slant_range_km / (111.0 * self.grid_size)) + 1)
        for di in range(-search_cells, search_cells + 1):
            for dj in range(-search_cells, search_cells + 1):
                cell = (center_cell[0] + di, center_cell[1] + dj)
                if cell in self._index:
                    candidate_indices.update(self._index[cell])

        # Filter by time overlap
        results = []
        for idx in candidate_indices:
            track = self.tracks[idx]
            if track.timestamps[-1] < time_start or track.timestamps[0] > time_end:
                continue
            results.append(track)

        return results


def parse_readsb_trace(
    data: dict,
    min_points: int = 5,
    target_time: float | None = None,
    window_s: float = 1800.0,
) -> FlightTrack | None:
    """
    Parse a readsb / adsb.lol / tar1090 trace dictionary into a FlightTrack.

    Expected format:
      {
        "icao": "c01234",
        "r": "C-GXYZ",
        "flight": "ACA123",
        "timestamp": 1780887600.0,
        "trace": [
          [time_offset, lat, lon, alt_baro, gs, track, geom_rate, ...],
          ...
        ]
      }
    """
    icao = data.get("icao", data.get("hex", "unknown"))
    callsign = (data.get("flight") or data.get("r") or "").strip()
    base_t = float(data.get("timestamp", 0.0))
    raw_trace = data.get("trace", [])

    if len(raw_trace) < min_points:
        return None

    # Fast time check on raw trace bounds if target_time is given
    if target_time is not None:
        try:
            first_t = base_t + float(raw_trace[0][0])
            last_t = base_t + float(raw_trace[-1][0])
            if last_t < target_time - window_s or first_t > target_time + window_s:
                return None
        except (IndexError, ValueError, TypeError):
            return None

    times, lats, lons, alts, vxs, vys, vzs = [], [], [], [], [], [], []
    for pt in raw_trace:
        if len(pt) < 3 or pt[1] is None or pt[2] is None:
            continue
        t_off = float(pt[0])
        abs_t = base_t + t_off

        # Filter strictly within the target time window to avoid interpolating 24h of data
        if target_time is not None and abs(abs_t - target_time) > window_s:
            continue

        lat = float(pt[1])
        lon = float(pt[2])

        # Altitude in ft -> meters
        raw_alt = pt[3] if len(pt) > 3 else None
        if raw_alt is not None and raw_alt != "ground":
            try:
                alt_m = float(raw_alt) * 0.3048
            except (ValueError, TypeError):
                alt_m = 3000.0
        else:
            alt_m = 0.0

        # Ground speed in knots -> m/s
        raw_gs = pt[4] if len(pt) > 4 else None
        try:
            speed_ms = float(raw_gs) * 0.514444 if raw_gs is not None else 0.0
        except (ValueError, TypeError):
            speed_ms = 0.0

        # Track heading in degrees
        raw_trk = pt[5] if len(pt) > 5 else None
        try:
            trk_rad = np.radians(float(raw_trk)) if raw_trk is not None else 0.0
        except (ValueError, TypeError):
            trk_rad = 0.0

        vx = speed_ms * np.sin(trk_rad)
        vy = speed_ms * np.cos(trk_rad)

        # Vertical rate in ft/min -> m/s
        raw_vr = pt[6] if len(pt) > 6 else None
        try:
            vz = float(raw_vr) * 0.00508 if raw_vr is not None else 0.0
        except (ValueError, TypeError):
            vz = 0.0

        times.append(abs_t)
        lats.append(lat)
        lons.append(lon)
        alts.append(alt_m)
        vxs.append(vx)
        vys.append(vy)
        vzs.append(vz)

    if len(times) < min_points:
        return None

    # Deduplicate strictly increasing timestamps
    times_arr = np.array(times)
    lats_arr = np.array(lats)
    lons_arr = np.array(lons)
    alts_arr = np.array(alts)
    vxs_arr = np.array(vxs)
    vys_arr = np.array(vys)
    vzs_arr = np.array(vzs)

    order = np.argsort(times_arr)
    times_arr = times_arr[order]
    _, unique_idx = np.unique(times_arr, return_index=True)

    if len(unique_idx) < min_points:
        return None

    t_u = times_arr[unique_idx]
    lat_u = lats_arr[order][unique_idx]
    lon_u = lons_arr[order][unique_idx]
    alt_u = alts_arr[order][unique_idx]
    vx_u = vxs_arr[order][unique_idx]
    vy_u = vys_arr[order][unique_idx]
    vz_u = vzs_arr[order][unique_idx]

    # Interpolate to 1-second grid only for the window
    t_interp = np.arange(t_u[0], t_u[-1] + 1.0, 1.0)
    lat_interp = np.interp(t_interp, t_u, lat_u)
    lon_interp = np.interp(t_interp, t_u, lon_u)
    alt_interp = np.interp(t_interp, t_u, alt_u)
    vx_interp = np.interp(t_interp, t_u, vx_u)
    vy_interp = np.interp(t_interp, t_u, vy_u)
    vz_interp = np.interp(t_interp, t_u, vz_u)

    return FlightTrack(
        icao24=str(icao).lower(),
        callsign=callsign,
        timestamps=t_interp,
        lat=lat_interp,
        lon=lon_interp,
        alt_m=alt_interp,
        vx=vx_interp,
        vy=vy_interp,
        vz=vz_interp,
    )


def download_github_traces(
    date: str,
    bbox: BoundingBox,
    target_time: float | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR / "github_archive",
) -> list[FlightTrack]:
    """
    Download and extract regional flight tracks from the adsblol/globe_history_YYYY
    GitHub release archives (supports 2023 onwards). Streams and extracts only flights
    passing within the specified bounding box, permanently caching them locally.
    """
    import gzip
    import json
    import tarfile
    import tempfile
    import urllib.request

    date_dir = cache_dir / date
    date_dir.mkdir(parents=True, exist_ok=True)

    # Check if we already have tracks in cache for this date
    cached_tracks = load_readsb_traces_dir(date_dir, bbox=bbox, target_time=target_time)
    if cached_tracks:
        logger.info("Loaded %d cached GitHub archive tracks from %s", len(cached_tracks), date_dir)
        return cached_tracks

    # Extract year from date string and format tag with dots (vYYYY.MM.DD)
    date_normalized = date.replace("/", "-").replace(".", "-")
    year = date_normalized.split("-")[0]
    repo_name = f"globe_history_{year}"
    date_dots = date_normalized.replace("-", ".")

    # adsblol tag naming: vYYYY.MM.DD-planes-readsb-prod-0
    tag = f"v{date_dots}-planes-readsb-prod-0"
    api_url = f"https://api.github.com/repos/adsblol/{repo_name}/releases/tags/{tag}"

    logger.info("Looking up GitHub ADS-B archive release (%s): %s", repo_name, tag)
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "skyhear/0.1"})
        with _safe_urlopen(req, timeout=10) as resp:
            release_data = json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning("Could not find GitHub archive for %s: %s", tag, exc)
        return []

    assets = release_data.get("assets", [])
    tar_assets = sorted([a for a in assets if ".tar." in a.get("name", "")], key=lambda a: a["name"])
    if not tar_assets:
        return []

    total_download_bytes = sum(a.get("size", 0) for a in tar_assets)
    total_mb = total_download_bytes / (1024 * 1024)
    import click
    click.echo(f"📦 Found daily ADS-B archive ({len(tar_assets)} parts, {total_mb:.1f} MB total).")

    matched_tracks: list[FlightTrack] = []

    # Stream download parts with clean single-line progress bar and extract matching planes
    with tempfile.NamedTemporaryFile(suffix=".tar") as temp_tar:
        with click.progressbar(
            length=total_download_bytes,
            label="⬇️  Downloading archive",
            width=30,
            show_percent=True,
            show_eta=True,
        ) as bar:
            last_render_bytes = 0
            # Update only every ~10 MB to prevent terminal scroll / newline spam
            update_threshold = 10 * 1024 * 1024
            for asset in tar_assets:
                download_url = asset.get("browser_download_url")
                try:
                    req_asset = urllib.request.Request(download_url, headers={"User-Agent": "skyhear/0.1"})
                    with _safe_urlopen(req_asset, timeout=120) as resp_asset:
                        while chunk := resp_asset.read(1024 * 1024):
                            temp_tar.write(chunk)
                            last_render_bytes += len(chunk)
                            if last_render_bytes >= update_threshold:
                                bar.update(last_render_bytes)
                                last_render_bytes = 0
                except Exception as exc:
                    logger.warning("Error streaming %s: %s", asset.get("name"), exc)
                    break
            if last_render_bytes > 0:
                bar.update(last_render_bytes)

        temp_tar.flush()
        temp_tar.seek(0)

        # Open and scan tar archive using generator iteration (O(1) memory)
        click.echo("🔎 Scanning and extracting regional flights for target airspace...")
        try:
            with tarfile.open(temp_tar.name, mode="r:*") as tar:
                # Iterate sequentially over tar members rather than loading all into memory
                processed_count = 0
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".json"):
                        continue
                    processed_count += 1
                    if processed_count % 5000 == 0:
                        click.echo(f"   Scanned {processed_count:,} trajectories, matched {len(matched_tracks)} flights...")

                    f = tar.extractfile(member)
                    if not f:
                        continue
                    try:
                        raw_bytes = f.read()
                        if raw_bytes[:2] == b"\x1f\x8b":
                            data = json.loads(gzip.decompress(raw_bytes))
                        else:
                            data = json.loads(raw_bytes)

                        track = parse_readsb_trace(data, target_time=target_time)
                        if track is None:
                            continue

                        # Check bounding box overlap (with acoustic horizon buffer)
                        margin_deg = 0.45
                        lat_in = (track.lat >= bbox.min_lat - margin_deg) & (track.lat <= bbox.max_lat + margin_deg)
                        lon_in = (track.lon >= bbox.min_lon - margin_deg) & (track.lon <= bbox.max_lon + margin_deg)
                        if not np.any(lat_in & lon_in):
                            continue

                        # Save to cache
                        dest_file = date_dir / f"trace_{track.icao24}.json"
                        with open(dest_file, "w", encoding="utf-8") as out_fp:
                            json.dump(data, out_fp)

                        matched_tracks.append(track)
                    except Exception:
                        continue
                    finally:
                        del raw_bytes
                        del data
                        tar.members.clear()
        except Exception as exc:
            logger.warning("Error reading tarball stream: %s", exc)
        except Exception as exc:
            logger.warning("Error reading tarball stream: %s", exc)

    click.echo(f"✅ Extracted and cached {len(matched_tracks)} flights matching target airspace.")
    return matched_tracks


def load_readsb_traces_dir(
    dir_path: str | Path,
    bbox: BoundingBox | None = None,
    target_time: float | None = None,
) -> list[FlightTrack]:
    """
    Load readsb / tar1090 / adsb.lol JSON trace files from a directory.

    Optionally filters by bounding box overlap and target time window.
    """
    import gzip
    import json

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        logger.warning("Trace directory does not exist: %s", dir_path)
        return []

    tracks = []
    # Search for json and json.gz files
    files = list(dir_path.glob("*.json")) + list(dir_path.glob("*.json.gz"))
    logger.info("Scanning %d trace files in %s", len(files), dir_path)

    for f in files:
        try:
            if f.suffix == ".gz":
                with gzip.open(f, "rt", encoding="utf-8") as fp:
                    data = json.load(fp)
            else:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            raw_trace = data.get("trace", [])
            if not raw_trace:
                continue

            if target_time is not None:
                base_t = float(data.get("timestamp", 0.0))
                try:
                    if (base_t + float(raw_trace[-1][0])) < target_time - 1800 or (base_t + float(raw_trace[0][0])) > target_time + 1800:
                        continue
                except (IndexError, ValueError, TypeError):
                    continue

            # Fast bounding box pre-filter on raw trace before expensive interpolation
            if bbox is not None:
                margin_deg = 0.45
                sample_pts = raw_trace[::max(1, len(raw_trace) // 15)]
                p_lats = [pt[1] for pt in sample_pts if len(pt) > 2 and pt[1] is not None]
                p_lons = [pt[2] for pt in sample_pts if len(pt) > 2 and pt[2] is not None]
                if p_lats and p_lons:
                    if (
                        max(p_lats) < bbox.min_lat - margin_deg
                        or min(p_lats) > bbox.max_lat + margin_deg
                        or max(p_lons) < bbox.min_lon - margin_deg
                        or min(p_lons) > bbox.max_lon + margin_deg
                    ):
                        continue

            track = parse_readsb_trace(data, target_time=target_time)
            if track is None:
                continue

            # Full bbox check on interpolated track
            if bbox is not None:
                margin_deg = 0.45
                lat_in = (track.lat >= bbox.min_lat - margin_deg) & (track.lat <= bbox.max_lat + margin_deg)
                lon_in = (track.lon >= bbox.min_lon - margin_deg) & (track.lon <= bbox.max_lon + margin_deg)
                if not np.any(lat_in & lon_in):
                    continue

            tracks.append(track)
        except Exception as exc:
            logger.debug("Error parsing %s: %s", f, exc)
        finally:
            del data

    logger.info("Loaded %d tracks from %s", len(tracks), dir_path)
    return tracks


def parse_opensky_track(data: dict, min_points: int = 5) -> FlightTrack | None:
    """
    Parse an OpenSky Network track response into a FlightTrack.
    Format of data['path']:
      [[time, lat, lon, baro_altitude, true_track, on_ground], ...]
    """
    path = data.get("path", [])
    if len(path) < min_points:
        return None

    icao = str(data.get("icao24", "")).lower()
    callsign = str(data.get("callsign", "") or "").strip()

    # Filter out ground and missing coordinates
    pts = [p for p in path if len(p) >= 5 and p[1] is not None and p[2] is not None and not p[5]]
    if len(pts) < min_points:
        return None

    times = np.array([float(p[0]) for p in pts])
    lats = np.array([float(p[1]) for p in pts])
    lons = np.array([float(p[2]) for p in pts])
    alts = np.array([float(p[3] if p[3] is not None else 3000.0) for p in pts])

    order = np.argsort(times)
    times = times[order]
    _, u_idx = np.unique(times, return_index=True)
    if len(u_idx) < min_points:
        return None

    times = times[u_idx]
    lats = lats[order][u_idx]
    lons = lons[order][u_idx]
    alts = alts[order][u_idx]

    lat_rad = np.radians(lats)
    dx = np.gradient(lons) * (111320.0 * np.cos(lat_rad))
    dy = np.gradient(lats) * 111320.0
    dt = np.gradient(times)
    dt = np.where(dt == 0, 1.0, dt)

    vx = dx / dt
    vy = dy / dt
    vz = np.gradient(alts) / dt

    # Interpolate to uniform 1-second grid
    t_interp = np.arange(times[0], times[-1] + 1.0, 1.0)
    lat_interp = np.interp(t_interp, times, lats)
    lon_interp = np.interp(t_interp, times, lons)
    alt_interp = np.interp(t_interp, times, alts)
    vx_interp = np.interp(t_interp, times, vx)
    vy_interp = np.interp(t_interp, times, vy)
    vz_interp = np.interp(t_interp, times, vz)

    return FlightTrack(
        icao24=icao,
        callsign=callsign,
        timestamps=t_interp,
        lat=lat_interp,
        lon=lon_interp,
        alt_m=alt_interp,
        vx=vx_interp,
        vy=vy_interp,
        vz=vz_interp,
    )


def fetch_live_tracks(
    bbox: BoundingBox,
    target_time: float | None = None,
    username: str | None = None,
    password: str | None = None,
) -> list[FlightTrack]:
    """
    Fetch live/recent ADS-B flight tracks (past 1 hour) for the given bounding box.
    Uses OpenSky Network and adsb.lol live snapshots, fetching full 4D flight paths
    in parallel for all active aircraft in the airspace.
    """
    import concurrent.futures
    import json
    import urllib.request

    center_lat = (bbox.min_lat + bbox.max_lat) / 2.0
    center_lon = (bbox.min_lon + bbox.max_lon) / 2.0
    radius_nm = int(max(abs(bbox.max_lat - bbox.min_lat), abs(bbox.max_lon - bbox.min_lon)) * 60.0 / 2.0) + 20

    candidate_icaos: set[str] = set()

    # 1. Query OpenSky live states
    opensky_url = (
        f"https://opensky-network.org/api/states/all?"
        f"lamin={bbox.min_lat:.4f}&lamax={bbox.max_lat:.4f}&"
        f"lomin={bbox.min_lon:.4f}&lomax={bbox.max_lon:.4f}"
    )
    try:
        req = urllib.request.Request(opensky_url, headers={"User-Agent": "skyhear/0.1"})
        with _safe_urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            states = data.get("states") or []
            for s in states:
                if s and s[0]:
                    candidate_icaos.add(str(s[0]).lower())
        logger.info("OpenSky live found %d aircraft in bounding box", len(states))
    except Exception as exc:
        logger.debug("OpenSky live query notice: %s", exc)

    # 2. Query adsb.lol live point
    adsb_url = f"https://api.adsb.lol/v2/lat/{center_lat:.4f}/lon/{center_lon:.4f}/dist/{radius_nm}"
    try:
        req2 = urllib.request.Request(adsb_url, headers={"User-Agent": "skyhear/0.1"})
        with _safe_urlopen(req2, timeout=8) as resp2:
            data2 = json.loads(resp2.read().decode())
            ac_list = data2.get("ac") or []
            for ac in ac_list:
                h = ac.get("hex")
                if h:
                    candidate_icaos.add(str(h).lower())
        logger.info("adsb.lol live found %d aircraft in radius %d nm", len(ac_list), radius_nm)
    except Exception as exc:
        logger.debug("adsb.lol query notice: %s", exc)

    if not candidate_icaos:
        logger.warning("No active aircraft found in the specified region via live feeds.")
        return []

    logger.info("Retrieving flight paths for %d active aircraft...", len(candidate_icaos))

    cache_dir = DEFAULT_CACHE_DIR / "live"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rate_limited = False

    def _fetch_one_track(icao: str) -> FlightTrack | None:
        nonlocal rate_limited
        cache_file = cache_dir / f"{icao}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return parse_opensky_track(json.load(f))
            except Exception:
                pass

        trk_url = f"https://opensky-network.org/api/tracks/all?icao24={icao}&time=0"
        try:
            req_headers = {"User-Agent": "skyhear/0.1"}
            if username and password:
                import base64
                auth_str = base64.b64encode(f"{username}:{password}".encode()).decode()
                req_headers["Authorization"] = f"Basic {auth_str}"
            req_trk = urllib.request.Request(trk_url, headers=req_headers)
            with _safe_urlopen(req_trk, timeout=6) as trk_resp:
                raw_json = trk_resp.read().decode()
                trk_data = json.loads(raw_json)
                track = parse_opensky_track(trk_data)
                if track is not None:
                    try:
                        with open(cache_file, "w", encoding="utf-8") as f:
                            f.write(raw_json)
                    except Exception:
                        pass
                return track
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                rate_limited = True
            return None
        except Exception:
            return None

    tracks: list[FlightTrack] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(_fetch_one_track, icao): icao for icao in candidate_icaos}
        for future in concurrent.futures.as_completed(future_map):
            trk = future.result()
            if trk is None:
                continue

            # Time filter if target_time provided (allow +/- 300s window)
            if target_time is not None:
                if trk.timestamps[-1] < target_time - 300 or trk.timestamps[0] > target_time + 300:
                    continue

            # Check bbox overlap (within 35 km)
            dlat = 35.0 / 111.0
            dlon = 35.0 / (111.0 * max(0.1, np.cos(np.radians(center_lat))))
            lat_ok = (trk.lat >= bbox.min_lat - dlat) & (trk.lat <= bbox.max_lat + dlat)
            lon_ok = (trk.lon >= bbox.min_lon - dlon) & (trk.lon <= bbox.max_lon + dlon)
            if not np.any(lat_ok & lon_ok):
                continue

            tracks.append(trk)

    if rate_limited and not tracks:
        if username:
            logger.warning(
                f"⚠️  OpenSky Network rate limit reached (HTTP 429) for account '{username}'. "
                "OpenSky daily quota exceeded. Falling back to live radar feeds..."
            )
        else:
            logger.warning(
                "⚠️  OpenSky Network rate-limited anonymous track requests (HTTP 429). "
                "To increase rate limits, register an account at opensky-network.org and set "
                "OPENSKY_USER in .env."
            )

    # Fallback to kinetic tracks from adsb.lol if OpenSky was rate-limited or returned 0 tracks
    if not tracks and ac_list:
        logger.info("Falling back to live radar vectors for active aircraft...")
        now_epoch = float(data2.get("now", 0)) / 1000.0 if "data2" in locals() and data2.get("now") else time.time()
        for ac in ac_list:
            lat = ac.get("lat")
            lon = ac.get("lon")
            alt_ft = ac.get("alt_baro")
            gs_kt = ac.get("gs")
            trk_deg = ac.get("track")
            seen = float(ac.get("seen_pos") or ac.get("seen") or 0.0)

            if lat is None or lon is None or alt_ft is None or alt_ft == "ground" or gs_kt is None or trk_deg is None:
                continue

            t0 = now_epoch - seen
            alt_m = float(alt_ft) * 0.3048
            speed_ms = float(gs_kt) * 0.514444
            heading_rad = np.radians(float(trk_deg))
            geom_rate = float(ac.get("geom_rate") or ac.get("baro_rate") or 0.0) * 0.00508

            vx = speed_ms * np.sin(heading_rad)
            vy = speed_ms * np.cos(heading_rad)
            vz = geom_rate

            ref_t = target_time if target_time is not None else t0
            t_start = min(ref_t - 300, t0 - 600)
            t_end = max(ref_t + 300, t0 + 600)
            t_interp = np.arange(t_start, t_end + 1.0, 1.0)
            dt = t_interp - t0

            lat_interp = lat + (vy * dt) / 111320.0
            lon_interp = lon + (vx * dt) / (111320.0 * np.cos(np.radians(lat)))
            alt_interp = np.maximum(0.0, alt_m + vz * dt)

            tracks.append(FlightTrack(
                icao24=str(ac.get("hex", "")).lower(),
                callsign=(ac.get("flight") or ac.get("r") or "").strip(),
                timestamps=t_interp,
                lat=lat_interp,
                lon=lon_interp,
                alt_m=alt_interp,
                vx=np.full_like(t_interp, vx),
                vy=np.full_like(t_interp, vy),
                vz=np.full_like(t_interp, vz),
            ))

    logger.info("Successfully acquired %d matching live flight tracks", len(tracks))
    return tracks


