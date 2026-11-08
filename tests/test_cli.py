"""
Tests for CLI options and date/live feed routing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from skyhear.cli import main


def test_cli_past_date_not_overwritten_by_file_metadata(tmp_path):
    runner = CliRunner()

    with patch("skyhear.ingest.extract_audio") as mock_extract, \
         patch("skyhear.ingest.load_audio") as mock_load, \
         patch("skyhear.detect.compute_spectrogram") as mock_spec, \
         patch("skyhear.detect.spectral_whitening"), \
         patch("skyhear.detect.auto_detect_ridges") as mock_ridges, \
         patch("skyhear.detect.plot_spectrogram"), \
         patch("skyhear.adsb.download_github_traces") as mock_dl, \
         patch("skyhear.adsb.fetch_live_tracks") as mock_live, \
         patch("skyhear.search.search_location") as mock_search, \
         patch("skyhear.report.calculate_confidence", return_value=85.0), \
         patch("skyhear.report.print_summary", return_value="Summary report"), \
         patch("skyhear.report.generate_results_json"), \
         patch("skyhear.report.plot_overlay"), \
         patch("av.open") as mock_av:

        # Mock media metadata with TODAY's timestamp
        mock_container = MagicMock()
        mock_container.metadata = {"creation_time": "2026-11-08T14:00:00Z"}
        mock_container.streams = []
        mock_av.return_value = mock_container

        mock_extract.return_value = MagicMock(duration_seconds=45.0, wav_path="/tmp/test.wav")
        mock_load.return_value = (MagicMock(), 22050)
        mock_spec.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_ridges.return_value = [MagicMock()]
        mock_dl.return_value = [MagicMock(icao24="abc123", timestamps=[0], lat=[0], lon=[0], alt_m=[1000])]

        cand = MagicMock(lat=34.0, lon=-118.0, alt_m=100.0, score=0.9, conf=90, estimated_f0=120.0, best_track_icao="abc123")
        mock_search.return_value = [cand]

        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"dummy")

        result = runner.invoke(
            main,
            [str(video_file), "--region", "bbox:30,-120,35,-115", "--date", "2024-05-15"],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"

        # download_github_traces must be called with the user's past date
        assert mock_dl.called
        called_date = mock_dl.call_args[0][0]
        assert called_date == "2024-05-15"

        # Live feeds must NOT have been called
        assert not mock_live.called


def test_skyhear_locate_library_api(tmp_path):
    import skyhear

    with patch("skyhear.locate.extract_audio") as mock_extract, \
         patch("skyhear.locate.load_audio") as mock_load, \
         patch("skyhear.locate.compute_spectrogram") as mock_spec, \
         patch("skyhear.locate.spectral_whitening"), \
         patch("skyhear.locate.auto_detect_ridges") as mock_ridges, \
         patch("skyhear.locate.download_github_traces") as mock_dl, \
         patch("skyhear.locate.search_location") as mock_search, \
         patch("av.open") as mock_av:

        mock_container = MagicMock()
        mock_container.metadata = {}
        mock_container.streams = []
        mock_av.return_value = mock_container

        mock_extract.return_value = MagicMock(duration_seconds=45.0, wav_path="/tmp/test.wav")
        mock_load.return_value = (MagicMock(), 22050)
        mock_spec.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_ridges.return_value = [MagicMock()]
        mock_dl.return_value = [MagicMock(icao24="abc123", timestamps=[0], lat=[0], lon=[0], alt_m=[1000])]

        cand = MagicMock(lat=34.0, lon=-118.0)
        mock_search.return_value = [cand]

        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"dummy")

        results = skyhear.locate(
            str(video_file),
            region="bbox:30,-120,35,-115",
            date="2024-05-15",
        )

        assert len(results) == 1
        assert results[0].lat == 34.0


def test_cli_rejects_audio_under_30_seconds(tmp_path):
    runner = CliRunner()

    with patch("skyhear.ingest.extract_audio") as mock_extract, \
         patch("av.open") as mock_av:
        mock_container = MagicMock()
        mock_container.metadata = {}
        mock_container.streams = []
        mock_av.return_value = mock_container

        mock_extract.return_value = MagicMock(duration_seconds=23.0, wav_path="/tmp/test.wav")

        video_file = tmp_path / "short.mp4"
        video_file.write_bytes(b"dummy")

        result = runner.invoke(
            main,
            [str(video_file), "--region", "bbox:30,-120,35,-115", "--date", "2024-05-15"],
        )

        assert result.exit_code != 0
        assert "Recording is too short (23.0s)" in result.output
        assert "Acoustic Doppler geolocation requires at least 30 seconds" in result.output


def test_locate_rejects_audio_under_30_seconds(tmp_path):
    import pytest
    import skyhear

    with patch("skyhear.locate.extract_audio") as mock_extract, \
         patch("av.open") as mock_av:
        mock_container = MagicMock()
        mock_container.metadata = {}
        mock_container.streams = []
        mock_av.return_value = mock_container

        mock_extract.return_value = MagicMock(duration_seconds=22.5, wav_path="/tmp/test.wav")

        video_file = tmp_path / "short.mp4"
        video_file.write_bytes(b"dummy")

        with pytest.raises(ValueError, match="Recording is too short"):
            skyhear.locate(
                str(video_file),
                region="bbox:30,-120,35,-115",
                date="2024-05-15",
            )


