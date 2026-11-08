# skyhear

**Aircraft-Noise Geolocation Tool** — Estimate where an audio/video recording was made by analyzing the faint sound of aircraft passing overhead and matching Doppler signatures against historical ADS-B flight data.

## How it works

![How Acoustic Doppler Aircraft Geolocation Works](docs/assets/how-it-works.jpg)

1. Jet aircraft produce tonal sounds (fan and compressor harmonics)
2. The **Doppler effect** shifts these frequencies as the aircraft approaches and recedes
3. The exact shape of the Doppler curve depends on the **receiver's ground position**
4. By matching observed curves against simulated flight paths from ADS-B radar data, the recording location can be triangulated to ~100–500 m

> [!NOTE]
> **Acoustic Environment & Range**:
> The system works best when audio is recorded outdoors or near an open window, where high-frequency engine tones (500–2500 Hz) are captured without wall dampening.
> 
> | Aircraft Altitude | Max Detection Range | Quality |
> |---|---|---|
> | **Low / Approach** (1,000–5,000 ft) | Up to **15–18 km** | **Strongest** — loud, distinct blade pass frequencies |
> | **Cruising** (25,000–38,000 ft) | Up to **10–15 km** (overhead cone) | **Good** — audible in low ambient noise |
> | **Distance > 25 km** | Beyond hearing threshold | **Inaudible** — high frequencies attenuate in the atmosphere |
> 
> *A single plane flyby constrains position to a ~5–10 km corridor; 2 or more crossing flights triangulate the location down to ~100–500 meters.*

---

## Installation

Requires **Python 3.12+** and **[ffmpeg](https://ffmpeg.org/)**.

```bash
git clone https://github.com/pooriababaei/skyhear.git
cd skyhear
pip install .
```

> **Tip**: You can also install directly without cloning:
> ```bash
> pip install git+https://github.com/pooriababaei/skyhear.git
> ```

---

## Usage

```bash
skyhear <file> --region <location> [options]
```

### Examples

```bash
# 1. Past recording with known date and time (recommended for fastest, most accurate match)
skyhear video.mp4 --region "Los Angeles" --date 2024-05-15 --start-time 14:35:00

# 2. Past recording with auto-detected date (read from video metadata)
skyhear video.mov --region "Seattle"

# 3. Live recording captured within the past hour
skyhear video.mp4 --region "Vancouver" --fresh

# 4. Search within a specific radius around coordinates
skyhear audio.wav --region "center:50.08,14.43,radius_km:50" --date 2024-06-10

# 5. Search within a custom bounding box (min_lat, min_lon, max_lat, max_lon)
skyhear audio.mp3 --region "bbox:49.0,-123.5,49.5,-122.5" --date 2024-05-15
```

---

## CLI Options

| Option | Description |
|---|---|
| `<file>` | Path to media file (`.mp4`, `.mov`, `.wav`, `.ogg`, `.m4a`, `.mp3`) |
| `-r`, `--region <name\|coord>` | Target search area: city name (`"Los Angeles"`), country (`"Germany"`), or coordinates (`"center:lat,lon,radius_km:N"`) |
| `-d`, `--date <YYYY-MM-DD>` | Recording date for historical flight lookups (auto-detected from file metadata if omitted) |
| `-t`, `--start-time <HH:MM[:SS]>` | Recording start time in regional local time or UTC |
| `--fresh` | Force real-time radar search for recordings captured within the last hour |
| `-o`, `--out <dir>` | Optional directory to save output files and plots (if omitted, results are only printed to terminal) |
| `-v`, `-vv` | Verbosity level (`-v` for info, `-vv` for debug) |

---

## Python Library Usage

You can also use `skyhear` directly inside your Python applications:

```python
import skyhear

# Locate an audio or video recording
candidates = skyhear.locate(
    "video.mp4",
    region="Los Angeles",
    date="2024-05-15",
    start_time="14:35:00",
)

for rank, cand in enumerate(candidates, 1):
    print(f"#{rank}: ({cand.lat:.6f}, {cand.lon:.6f})")
```

---

## Data Sources & Performance

- **Historical Mode (`--date`)**: Automatically streams and extracts true ground-radar traces from open [ADS-B daily archives](https://github.com/adsblol) (supported from 2023 up to yesterday's flights via `globe_history_YYYY`), with automatic fallback to [OpenSky Network](https://opensky-network.org/) if the daily archive is not yet published. The first query for a date downloads and indexes the airspace in **~15–30 seconds**; subsequent runs for that date are **instant (0s)** from local disk cache (`~/.skyhear/adsb_cache/`).
- **Live Mode (`--fresh`)**: Queries active regional ADS-B receivers via `adsb.lol` and OpenSky for recordings made within the past hour. Returns in **~1–2 seconds**.

---

## Pipeline Architecture

```
┌─────────┐    ┌──────────┐    ┌────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
│ ingest  │ →  │ detect   │ →  │ adsb   │ →  │ model   │ →  │ search   │ →  │ report   │
│         │    │          │    │        │    │         │    │          │    │          │
│ Extract │    │ STFT +   │    │ Fetch  │    │ Doppler │    │ Grid     │    │ JSON +   │
│ audio   │    │ ridges   │    │ flights│    │ sim     │    │ search   │    │ plots    │
└─────────┘    └──────────┘    └────────┘    └─────────┘    └──────────┘    └──────────┘
```

## Legal & Ethical Use

- Test only on your own recordings or recordings you have permission to analyze.
- Respect ADS-B data providers' terms of service and rate limits.
- Designed for acoustic geolocation, OSINT verification, and academic research.

## License

MIT
