from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_bird_db import xeno_canto_audio


class XenoCantoApiWorkflowTests(unittest.TestCase):
    def test_load_api_key_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env_file = root / "xeno_canto_api_key.env"
            env_file.write_text("export XENO_CANTO_API_KEY=secret-key\n", encoding="utf-8")
            fake_paths = type("P", (), {"root": root})
            with patch.dict(os.environ, {}, clear=True), patch.object(xeno_canto_audio, "get_project_paths", return_value=fake_paths):
                self.assertEqual(xeno_canto_audio.load_xeno_canto_api_key(None), "secret-key")

    def test_api_query_and_url(self) -> None:
        query = xeno_canto_audio.api_query_for_species_id("Corvus-macrorhynchos")
        self.assertEqual(query, 'sp:"Corvus macrorhynchos" q:A')
        url = xeno_canto_audio.api_recordings_url("Corvus-macrorhynchos", api_key="demo", page=2, per_page=50)
        self.assertIn("https://xeno-canto.org/api/3/recordings?", url)
        self.assertIn("query=sp%3A%22Corvus%20macrorhynchos%22%20q%3AA", url)
        self.assertIn("page=2", url)
        self.assertIn("per_page=50", url)
        self.assertIn("key=demo", url)

    def test_api_fetch_recordings_and_recording_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_tsv = root / "bird_xeno_canto_ids.tsv"
            with input_tsv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, delimiter="\t", fieldnames=["qid", "xeno_canto_species_id"])
                writer.writeheader()
                writer.writerow({"qid": "Q122868", "xeno_canto_species_id": "Corvus-macrorhynchos"})

            api_output_dir = root / "api_recordings"
            page_1_url = xeno_canto_audio.api_recordings_url("Corvus-macrorhynchos", api_key="demo", page=1, per_page=100)
            page_2_url = xeno_canto_audio.api_recordings_url("Corvus-macrorhynchos", api_key="demo", page=2, per_page=100)
            payloads = {
                page_1_url: {
                    "numPages": 2,
                    "recordings": [
                        {"id": "111", "file": "https://xeno-canto.org/111/download", "uploaded": "2025-05-02"}
                    ],
                },
                page_2_url: {
                    "numPages": 2,
                    "recordings": [
                        {"id": "112", "file": "https://xeno-canto.org/112/download", "uploaded": "2025-05-03"}
                    ],
                },
            }

            result = xeno_canto_audio.fetch_xeno_canto_recording_jsons(
                input_path=input_tsv,
                output_dir=api_output_dir,
                api_key="demo",
                per_page=100,
                max_pages=5,
                sleep_seconds=0,
                fetch_json_fn=lambda url: payloads[url],
            )

            summary_path = api_output_dir / "api_recordings_summary.json"
            self.assertTrue(summary_path.exists())
            self.assertEqual(result["summary"]["target_qid_count"], 1)

            page_1_path = api_output_dir / "Q122868" / "page001.json"
            page_2_path = api_output_dir / "Q122868" / "page002.json"
            self.assertTrue(page_1_path.exists())
            self.assertTrue(page_2_path.exists())

            recording_map_json = root / "recording_map.json"
            map_result = xeno_canto_audio.build_xeno_canto_recording_map_from_api(
                input_path=api_output_dir,
                output_json=recording_map_json,
            )
            self.assertTrue(recording_map_json.exists())
            self.assertEqual(map_result["summary"]["recording_count"], 2)

            recording_map = json.loads(recording_map_json.read_text(encoding="utf-8"))
            self.assertEqual(recording_map[0]["recording_ids"], ["111", "112"])
            self.assertEqual(
                recording_map[0]["download_urls"],
                ["https://xeno-canto.org/111/download", "https://xeno-canto.org/112/download"],
            )

            audio_output_dir = root / "audio"
            downloaded_urls: list[str] = []

            def fake_download_bytes(url: str) -> bytes:
                downloaded_urls.append(url)
                return f"payload:{url}".encode("utf-8")

            def fake_clip_audio(tmp_input: Path, local_path: Path, file_type: str, clip_seconds: int) -> None:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(f"{file_type}:{clip_seconds}:{tmp_input.read_text(encoding='utf-8')}".encode("utf-8"))

            audio_result = xeno_canto_audio.fetch_audio_from_recording_map(
                input_path=recording_map_json,
                output_dir=audio_output_dir,
                limit_per_qid=10,
                clip_seconds=30,
                sleep_seconds=0,
                download_bytes_fn=fake_download_bytes,
                clip_audio_fn=fake_clip_audio,
            )

            self.assertEqual(downloaded_urls, ["https://xeno-canto.org/111/download", "https://xeno-canto.org/112/download"])
            self.assertTrue((audio_output_dir / "Q122868" / "111.mp3").exists())
            self.assertTrue((audio_output_dir / "Q122868" / "112.mp3").exists())
            self.assertTrue((audio_output_dir / "existing_audio_manifest.json").exists())
            self.assertEqual(audio_result["downloaded_qid_count"], 1)
            self.assertEqual(audio_result["status_rows"][0]["selected_count"], 2)

            second_download_calls: list[str] = []

            def fail_if_downloaded(url: str) -> bytes:
                second_download_calls.append(url)
                raise AssertionError("download should have been skipped for existing files")

            second_result = xeno_canto_audio.fetch_audio_from_recording_map(
                input_path=recording_map_json,
                output_dir=audio_output_dir,
                limit_per_qid=10,
                clip_seconds=30,
                sleep_seconds=0,
                download_bytes_fn=fail_if_downloaded,
                clip_audio_fn=fake_clip_audio,
            )

            self.assertEqual(second_download_calls, [])
            self.assertEqual(second_result["status_rows"][0]["selected_count"], 2)


if __name__ == "__main__":
    unittest.main()
