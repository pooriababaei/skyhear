# skyhear

**Aircraft-Noise Geolocation Tool** — Estimate where an audio/video recording was made by analyzing the faint sound of aircraft passing overhead and matching Doppler signatures against historical ADS-B flight data.

## How it works

![How Acoustic Doppler Aircraft Geolocation Works](docs/assets/how-it-works.jpg)

1. Jet aircraft produce tonal sound (fan/compressor harmonics)
2. The **Doppler effect** bends these tones as the plane approaches and recedes
3. The exact shape of the Doppler curve depends on the **receiver's ground position**
4. By matching observed curves against simulated ones (using ADS-B flight tracks), the recording location can be estimated to ~100–500 m

> [!NOTE]
> **Recording environment & aircraft proximity**:
> The system works best when audio is recorded **outdoors or near an open window**, where high-frequency engine blade tones (typically 500–2500 Hz) are captured without wall dampening.
> 
> | Aircraft Altitude | Max Detection Range | Quality |
> |---|---|---|
> | **Low / Approach** (1,000–5,000 ft) | Up to **15–18 km** | **Strongest** — loud, distinct blade pass frequencies |
> | **Cruising** (25,000–38,000 ft) | Up to **10–15 km** (overhead cone) | **Good** — audible in low ambient noise |
> | **Distance > 25 km** | Beyond hearing threshold | **Inaudible** — high frequencies attenuate in the atmosphere |
> 
> *A single plane flyby constrains position to a ~5–10 km corridor; 2 or more crossing flights triangulate the location down to ~100–500 meters.*

## Installation

Requires **Python 3.12+** and **ffmpeg** installed on your system.

### Recommended: Virtual Environment Setup

Using a dedicated virtual environment ensures all audio processing and scientific dependencies are isolated, and makes the `skyhear` CLI command immediately available without PATH issues.

#### **macOS / Linux:**
```bash
# 1. Clone the repository
git clone https://github.com/pooriababaei/skyhear.git
cd skyhear

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install skyhear in editable mode
pip install -e .
```

#### **Windows (PowerShell):**
```powershell
# 1. Clone the repository
git clone https://github.com/pooriababaei/skyhear.git
cd skyhear

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install skyhear in editable mode
pip install -e .
```
*(If PowerShell script execution is restricted, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

---

### Running Directly Without Activating

You can also run commands directly through the virtual environment binary or Python's module runner without needing to activate:

- **macOS / Linux:**
  ```bash
  ./.venv/bin/skyhear clip.mp4 --region "London" --date 2026-06-07
  # OR
  python3 -m skyhear clip.mp4 --region "London" --date 2026-06-07
  ```

- **Windows:**
  ```powershell
  .\.venv\Scripts\skyhear.exe clip.mp4 --region "Prague" --date 2026-06-07
  # OR
  python -m skyhear clip.mp4 --region "Prague" --date 2026-06-07
  ```

## Usage

```bash
# Locate a fresh recording made in the past hour
skyhear clip.mp4 --region "Prague" --fresh

# Locate a recording with a known calendar date
skyhear clip.mp4 --region "Czech Republic" --date 2026-06-07

# Locate using an explicit center coordinate and search radius
skyhear clip.mp4 --region "center:49.80,15.47,radius_km:200" --date 2026-06-07
```

### Options

| Flag | Description |
|------|-------------|
| `<file>` or `--input`, `-i` | Input media file (mp4, mov, wav, ogg, m4a, mp3, etc.) |
| `--region`, `-r` | Search region: city name (`"London"`), country/state (`"Czech Republic"`), or `center:lat,lon,radius_km:N` |
| `--fresh` | Use live radar data for recordings made in the past hour (no date needed) |
| `--date`, `-d` | Recording date (`YYYY-MM-DD`) for historical recordings |
| `--opensky-user` | *(Optional)* OpenSky Network username (or `OPENSKY_USER` env var) |
| `--opensky-pass` | *(Optional)* OpenSky Network password (or `OPENSKY_PASS` env var) |

### Data Sources & Speed

- **`--date <YYYY-MM-DD>`**: Automatically streams and extracts true ground-radar traces from open [ADS-B daily archives](https://github.com/adsblol) (supported from 2023 up to yesterday's flights via `globe_history_YYYY`), with automatic fallback to [OpenSky Network](https://opensky-network.org/) if that date's archive is not yet published. For any given date, the first run takes **~15–30 seconds** (extracting regional flights), and subsequent runs for that same date are **instant (0s)** from local disk cache (`~/.skyhear/adsb_cache/`).
- **`--fresh`**: Pings live regional ADS-B feeds in real time via `adsb.lol` for recordings made within the past hour. Takes **~1–2 seconds** to fetch.

> [!NOTE]
> Due to [OpenSky Network API limitations](https://openskynetwork.github.io/opensky-api/rest.html) (tracks cut off after 30 days), `skyhear` uses permanent daily GitHub archives so queries are never cut off.

## Pipeline

```
┌─────────┐    ┌──────────┐    ┌────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
│ ingest  │ →  │ detect   │ →  │ adsb   │ →  │ model   │ →  │ search   │ →  │ report   │
│         │    │          │    │        │    │         │    │          │    │          │
│ Extract │    │ STFT +   │    │ Fetch  │    │ Doppler │    │ Grid     │    │ JSON +   │
│ audio   │    │ ridges   │    │ flights│    │ sim     │    │ search   │    │ plots    │
└─────────┘    └──────────┘    └────────┘    └─────────┘    └──────────┘    └──────────┘
```

## Legal / Ethical

- Test only on your own recordings or recordings you have permission to analyze
- Respect ADS-B data providers' terms of service and rate limits
- Purpose: privacy research and education

## License

MIT
