from __future__ import annotations

import pickle
import tempfile
import unittest
from datetime import date
from pathlib import Path

from multi_bird_db.xeno_canto_audio import (
    extract_recording_ids,
    fetch_qid_audio,
    fetch_xeno_canto_audio,
    parse_recording_page,
)


class XenoCantoAudioTests(unittest.TestCase):
    def test_extract_recording_ids_preserves_order_and_deduplicates(self) -> None:
        html = """
        <table>
          <tr><td>XC123456</td></tr>
          <tr><td>XC123456</td></tr>
          <tr><td>XC654321</td></tr>
        </table>
        """
        self.assertEqual(extract_recording_ids(html), ["XC123456", "XC654321"])

    def test_parse_recording_page_reads_uploaded_and_file_type(self) -> None:
        html = """
        <table>
          <tr><td>Uploaded  |  2025-06-07</td></tr>
          <tr><td>File type  |  wav</td></tr>
        </table>
        """
        self.assertEqual(parse_recording_page(html), {"uploaded": "2025-06-07", "file_type": "wav"})

    def test_fetch_qid_audio_filters_by_upload_date(self) -> None:
        species_html = "<div>XC1001 XC1002 XC1003</div>"
        recording_pages = {
            "https://xeno-canto.org/species/species-id?order=rec&pg=1": species_html,
            "https://xeno-canto.org/1001": "<table><tr><td>Uploaded | 2025-05-02</td></tr><tr><td>File type | mp3</td></tr></table>",
            "https://xeno-canto.org/1002": "<table><tr><td>Uploaded | 2025-04-30</td></tr><tr><td>File type | mp3</td></tr></table>",
            "https://xeno-canto.org/1003": "<table><tr><td>Uploaded | 2025-06-01</td></tr><tr><td>File type | wav</td></tr></table>",
        }
        downloaded: list[str] = []

        def fetch_text(url: str) -> str:
            return recording_pages[url]

        def download_bytes(url: str) -> bytes:
            downloaded.append(url)
            return f"payload:{url}".encode("utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            rows, status = fetch_qid_audio(
                qid="Q1",
                xeno_canto_species_id="species-id",
                output_dir=Path(tmpdir),
                since_date=date(2025, 5, 1),
                limit_per_qid=10,
                max_pages=1,
                sleep_seconds=0.0,
                fetch_text_fn=fetch_text,
                download_bytes_fn=download_bytes,
            )

            self.assertEqual(status["status"], "ok")
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["recording_id"] for row in rows], ["XC1001", "XC1003"])
            self.assertEqual(downloaded, ["https://xeno-canto.org/1001/download", "https://xeno-canto.org/1003/download"])
            self.assertTrue((Path(tmpdir) / "Q1" / "XC1001.mp3").exists())
            self.assertTrue((Path(tmpdir) / "Q1" / "XC1003.wav").exists())

    def test_fetch_xeno_canto_audio_writes_root_manifests(self) -> None:
        ontology_rows = [{"qid": "Q1", "xeno_canto_species_id": "species-id"}]
        species_html = "<div>XC1001</div>"
        recording_pages = {
            "https://xeno-canto.org/species/species-id?order=rec&pg=1": species_html,
            "https://xeno-canto.org/1001": "<table><tr><td>Uploaded | 2025-05-02</td></tr><tr><td>File type | mp3</td></tr></table>",
        }

        def fetch_text(url: str) -> str:
            return recording_pages[url]

        def download_bytes(_: str) -> bytes:
            return b"audio"

        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
                try:
                    pickle.dump(ontology_rows, handle)
                    handle.flush()
                    ontology_path = Path(handle.name)
                finally:
                    handle.close()
            try:
                output_dir = Path(tmpdir)
                result = fetch_xeno_canto_audio(
                    ontology_path=ontology_path,
                    output_dir=output_dir,
                    since_date=date(2025, 5, 1),
                    limit_per_qid=10,
                    max_pages=1,
                    sleep_seconds=0.0,
                    fetch_text_fn=fetch_text,
                    download_bytes_fn=download_bytes,
                )

                self.assertEqual(result["summary"]["recording_count"], 1)
                self.assertTrue((output_dir / "audio_manifest.tsv").exists())
                self.assertTrue((output_dir / "audio_ids.json").exists())
                self.assertTrue((output_dir / "qids.json").exists())
                self.assertTrue((output_dir / "metadata.json").exists())
                self.assertTrue((output_dir / "summary.json").exists())
                self.assertTrue((output_dir / "Q1" / "XC1001.mp3").exists())
            finally:
                ontology_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
