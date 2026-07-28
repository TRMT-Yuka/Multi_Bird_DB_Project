from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from multi_bird_db import audio_embeddings


class AudioEmbeddingTests(unittest.TestCase):

    def test_resolve_media_binary_uses_explicit_ffmpeg_bin_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            binary = Path(tmp_dir) / "ffmpeg"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o755)
            with (
                mock.patch.object(audio_embeddings.shutil, "which", return_value=None),
                mock.patch.dict(audio_embeddings.os.environ, {audio_embeddings.FFMPEG_BIN_DIR_ENV: tmp_dir}),
                mock.patch.object(audio_embeddings.Path, "home", return_value=Path("/does/not/exist")),
            ):
                self.assertEqual(audio_embeddings._resolve_media_binary("ffmpeg"), str(binary))

    def test_resolve_media_binary_uses_static_home_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            static_dir = home / "ffmpeg-7.0.2-arm64-static"
            static_dir.mkdir()
            binary = static_dir / "ffprobe"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o755)
            with (
                mock.patch.object(audio_embeddings.shutil, "which", return_value=None),
                mock.patch.dict(audio_embeddings.os.environ, {}, clear=True),
                mock.patch.object(audio_embeddings.Path, "home", return_value=home),
            ):
                self.assertEqual(audio_embeddings._resolve_media_binary("ffprobe"), str(binary))

    def test_resolve_birdnet_runtime_uses_tf_on_cpu_without_tensorflow_gpu(self) -> None:
        with mock.patch.object(audio_embeddings, "birdnet_gpu_available", return_value=False):
            backend, runtime_device = audio_embeddings.resolve_birdnet_runtime("auto")
        self.assertEqual(backend, "tf")
        self.assertEqual(runtime_device, "CPU")

    def test_resolve_birdnet_runtime_uses_pb_on_gpu(self) -> None:
        with mock.patch.object(audio_embeddings, "birdnet_gpu_available", return_value=True):
            backend, runtime_device = audio_embeddings.resolve_birdnet_runtime("cuda")
        self.assertEqual(backend, "pb")
        self.assertEqual(runtime_device, "GPU:0")

    def test_birdnet_encoder_uses_real_batching_on_gpu(self) -> None:
        structured_dtype = np.dtype(
            [
                ("input", "U256"),
                ("start_time", np.float32),
                ("end_time", np.float32),
                ("embedding", np.float32, (2,)),
            ]
        )

        class FakeStructuredResult:
            def __init__(self, rows):
                self._rows = rows

            def to_structured_array(self):
                return self._rows

        class FakeBirdNETModel:
            def __init__(self):
                self.last_kwargs = None

            def encode(self, paths, **kwargs):
                self.last_kwargs = {"paths": list(paths), **kwargs}
                rows = np.zeros(len(paths), dtype=structured_dtype)
                for index, path in enumerate(paths):
                    rows[index] = (str(path), 0.0, 3.0, np.asarray([float(index), float(index + 1)], dtype=np.float32))
                return FakeStructuredResult(rows)

        fake_model = FakeBirdNETModel()
        with (
            mock.patch.object(audio_embeddings, "birdnet_gpu_available", return_value=True),
            mock.patch.object(audio_embeddings.mp, "get_start_method", return_value=None),
            mock.patch.object(audio_embeddings.mp, "set_start_method") as mock_set_start_method,
            mock.patch.object(audio_embeddings.os, "cpu_count", return_value=16),
        ):
            encoder = audio_embeddings.BirdNETAudioEncoder(backend="auto", device="cuda", model=fake_model, model_batch_size=8)
            structured = encoder.encode_files([Path("a.mp3"), Path("b.mp3"), Path("c.mp3")])

        self.assertEqual(structured.shape[0], 3)
        mock_set_start_method.assert_called_once_with("spawn", force=True)
        assert fake_model.last_kwargs is not None
        self.assertEqual(fake_model.last_kwargs["device"], "GPU:0")
        self.assertEqual(fake_model.last_kwargs["batch_size"], 3)
        self.assertEqual(fake_model.last_kwargs["n_workers"], 1)
        self.assertEqual(fake_model.last_kwargs["n_producers"], 3)

    def test_birdnet_gpu_path_forces_spawn_and_single_worker(self) -> None:
        class FakeBirdNETModel:
            def __init__(self):
                self.last_kwargs = None

            def encode(self, paths, **kwargs):
                self.last_kwargs = kwargs
                structured_dtype = np.dtype(
                    [
                        ("input", "U256"),
                        ("start_time", np.float32),
                        ("end_time", np.float32),
                        ("embedding", np.float32, (2,)),
                    ]
                )
                rows = np.zeros(len(paths), dtype=structured_dtype)
                for index, path in enumerate(paths):
                    rows[index] = (str(path), 0.0, 3.0, np.asarray([1.0, 2.0], dtype=np.float32))

                class FakeStructuredResult:
                    def __init__(self, rows):
                        self._rows = rows

                    def to_structured_array(self):
                        return self._rows

                return FakeStructuredResult(rows)

        fake_model = FakeBirdNETModel()
        with (
            mock.patch.object(audio_embeddings, "birdnet_gpu_available", return_value=True),
            mock.patch.object(audio_embeddings.mp, "get_start_method", return_value=None),
            mock.patch.object(audio_embeddings.mp, "set_start_method") as mock_set_start_method,
        ):
            encoder = audio_embeddings.BirdNETAudioEncoder(backend="auto", device="cuda", model=fake_model)
            matrix = encoder.encode_files([Path("a.mp3")])

        self.assertEqual(matrix.shape[0], 1)
        mock_set_start_method.assert_called_once_with("spawn", force=True)
        assert fake_model.last_kwargs is not None
        self.assertEqual(fake_model.last_kwargs["device"], "GPU:0")
        self.assertEqual(fake_model.last_kwargs["batch_size"], 1)
        self.assertEqual(fake_model.last_kwargs["n_workers"], 1)
        self.assertEqual(fake_model.last_kwargs["n_producers"], 2)

    def test_discover_and_embed_audio_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "trimmed_xeno_data"
            qid_dir = input_dir / "Q122868"
            qid_dir.mkdir(parents=True)
            (qid_dir / "288057.mp3").write_bytes(b"stub-audio-one")
            (qid_dir / "119100.mp3").write_bytes(b"stub-audio-two")

            decoded_lengths: list[int] = []

            def fake_loader(path: Path) -> tuple[np.ndarray, int]:
                payload = path.read_bytes()
                decoded_lengths.append(len(payload))
                return np.arange(len(payload), dtype=np.float32), 16000

            class FakeEncoder:
                sample_rate = 16000

                def encode_batch(self, waveforms):
                    rows = []
                    for waveform in waveforms:
                        rows.append(
                            np.array(
                                [
                                    float(waveform.shape[0]),
                                    float(waveform[0] if waveform.shape[0] else 0.0),
                                    float(np.mean(waveform) if waveform.shape[0] else 0.0),
                                ],
                                dtype=np.float32,
                            )
                        )
                    return np.vstack(rows)

            output_dir = root / "embeddings"
            result = audio_embeddings.build_audio_embeddings(
                input_dir=input_dir,
                output_dir=output_dir,
                model_name="unit-test-model",
                device="cpu",
                batch_size=1,
                max_seconds=30.0,
                target_sample_rate=16000,
                extensions=("mp3",),
                cache_dir=None,
                audio_loader=fake_loader,
                encoder=FakeEncoder(),
            )

            run_dir = Path(result["summary"]["run_dir"])
            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "embeddings.npy").exists())
            self.assertTrue((run_dir / "audio_ids.json").exists())
            self.assertTrue((run_dir / "qids.json").exists())
            self.assertTrue((run_dir / "audio_manifest.tsv").exists())
            self.assertTrue((run_dir / "metadata.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "failed_items.json").exists())
            self.assertTrue((run_dir / "embeddings.partial.npy").exists())
            self.assertTrue((run_dir / "audio_ids.partial.json").exists())
            self.assertTrue((run_dir / "qids.partial.json").exists())
            self.assertTrue((run_dir / "metadata.partial.json").exists())
            self.assertTrue((run_dir / "audio_manifest.partial.tsv").exists())
            self.assertTrue((run_dir / "failed_items.partial.json").exists())
            self.assertTrue((run_dir / "summary.partial.json").exists())

            embeddings = np.load(run_dir / "embeddings.npy")
            self.assertEqual(embeddings.shape, (2, 3))
            audio_ids = json.loads((run_dir / "audio_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(audio_ids, ["Q122868_119100", "Q122868_288057"])
            qids = json.loads((run_dir / "qids.json").read_text(encoding="utf-8"))
            self.assertEqual(qids, ["Q122868", "Q122868"])
            self.assertEqual(result["summary"]["item_count"], 2)
            self.assertEqual(result["summary"]["failed_count"], 0)
            self.assertEqual(len(decoded_lengths), 2)
            self.assertEqual(result["store"].metadata["decoder"], "ffmpeg_or_custom_loader")

            manifest_rows = result["manifest_rows"]
            self.assertEqual(len(manifest_rows), 2)
            self.assertEqual(manifest_rows[0]["embedding_index"], "0")
            self.assertEqual(manifest_rows[1]["embedding_index"], "1")

    def test_birdnet_backend_uses_three_second_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "trimmed_xeno_data"
            qid_dir = input_dir / "Q122868"
            qid_dir.mkdir(parents=True)
            (qid_dir / "111.mp3").write_bytes(b"stub-audio-one")
            (qid_dir / "222.mp3").write_bytes(b"stub-audio-two")

            def fake_loader(path: Path) -> tuple[np.ndarray, int]:
                if path.name == "111.mp3":
                    return np.linspace(-1.0, 1.0, 5 * 48000, dtype=np.float32), 48000
                return np.linspace(-0.5, 0.5, 2 * 48000, dtype=np.float32), 48000

            class FakeBirdNETEncoder:
                sample_rate = 48000
                model_type = "acoustic"
                model_version = "2.4"
                backend = "tf"

                def encode_files(self, paths, *, max_audio_duration_min=None):
                    structured_dtype = np.dtype(
                        [
                            ("input", "U256"),
                            ("start_time", np.float32),
                            ("end_time", np.float32),
                            ("embedding", np.float32, (3,)),
                        ]
                    )
                    rows = np.zeros(3, dtype=structured_dtype)
                    rows[0] = (str(paths[0]), 0.0, 3.0, np.asarray([0.0, 5.0, 0.0], dtype=np.float32))
                    rows[1] = (str(paths[0]), 3.0, 5.0, np.asarray([1.0, 5.0, 0.0], dtype=np.float32))
                    rows[2] = (str(paths[1]), 0.0, 2.0, np.asarray([2.0, 2.0, 0.0], dtype=np.float32))
                    return rows

            output_dir = root / "embeddings"
            result = audio_embeddings.build_audio_embeddings(
                input_dir=input_dir,
                output_dir=output_dir,
                backend="birdnet",
                model_name="birdnet-unit-test",
                device="cpu",
                batch_size=2,
                max_seconds=10.0,
                target_sample_rate=48000,
                extensions=("mp3",),
                cache_dir=None,
                encoder=FakeBirdNETEncoder(),
            )

            run_dir = Path(result["summary"]["run_dir"])
            embeddings = np.load(run_dir / "embeddings.npy")
            self.assertEqual(embeddings.shape, (3, 3))

            audio_ids = json.loads((run_dir / "audio_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(audio_ids, ["Q122868_111_w0000", "Q122868_111_w0001", "Q122868_222_w0000"])

            manifest_rows = result["manifest_rows"]
            self.assertEqual(manifest_rows[0]["window_index"], "0")
            self.assertEqual(manifest_rows[0]["window_seconds"], "3.000000")
            self.assertEqual(manifest_rows[1]["window_index"], "1")
            self.assertEqual(manifest_rows[2]["window_index"], "0")
            self.assertEqual(result["summary"]["item_count"], 3)
            self.assertEqual(result["summary"]["failed_count"], 0)

    def test_birdnet_backend_uses_official_file_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "trimmed_xeno_data"
            qid_dir = input_dir / "Q122868"
            qid_dir.mkdir(parents=True)
            file_one = qid_dir / "111.mp3"
            file_two = qid_dir / "222.mp3"
            file_one.write_bytes(b"stub-audio-one")
            file_two.write_bytes(b"stub-audio-two")

            structured_dtype = np.dtype(
                [
                    ("input", "U256"),
                    ("start_time", np.float32),
                    ("end_time", np.float32),
                    ("embedding", np.float32, (3,)),
                ]
            )

            class FakeBirdNETEncoder:
                device = "cuda"
                sample_rate = 48000
                model_type = "acoustic"
                model_version = "2.4"
                backend = "pb"

                def __init__(self):
                    self.calls = []

                def encode_files(self, paths, *, max_audio_duration_min=None):
                    self.calls.append(([str(path) for path in paths], max_audio_duration_min))
                    rows = np.zeros(3, dtype=structured_dtype)
                    rows[0] = (str(file_one), 0.0, 3.0, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
                    rows[1] = (str(file_one), 3.0, 6.0, np.asarray([4.0, 5.0, 6.0], dtype=np.float32))
                    rows[2] = (str(file_two), 0.0, 3.0, np.asarray([7.0, 8.0, 9.0], dtype=np.float32))
                    return rows

            fake_encoder = FakeBirdNETEncoder()
            output_dir = root / "embeddings"
            result = audio_embeddings.build_audio_embeddings(
                input_dir=input_dir,
                output_dir=output_dir,
                backend="birdnet",
                model_name="birdnet-unit-test",
                device="cuda",
                batch_size=8,
                max_seconds=30.0,
                target_sample_rate=48000,
                extensions=("mp3",),
                cache_dir=None,
                encoder=fake_encoder,
            )

            run_dir = Path(result["summary"]["run_dir"])
            embeddings = np.load(run_dir / "embeddings.npy")
            self.assertEqual(embeddings.shape, (3, 3))

            audio_ids = json.loads((run_dir / "audio_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(audio_ids, ["Q122868_111_w0000", "Q122868_111_w0001", "Q122868_222_w0000"])

            manifest_rows = result["manifest_rows"]
            self.assertEqual(manifest_rows[0]["window_index"], "0")
            self.assertEqual(manifest_rows[0]["window_start_seconds"], "0.000000")
            self.assertEqual(manifest_rows[1]["window_index"], "1")
            self.assertEqual(manifest_rows[1]["window_start_seconds"], "3.000000")
            self.assertEqual(manifest_rows[2]["window_index"], "0")
            self.assertEqual(result["summary"]["item_count"], 3)
            self.assertEqual(result["summary"]["failed_count"], 0)
            self.assertEqual(result["store"].metadata["decoder"], "birdnet_file_api")
            self.assertEqual(fake_encoder.calls, [([str(file_one), str(file_two)], None)])

    def test_birdnet_backend_resumes_from_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "trimmed_xeno_data"
            qid_dir = input_dir / "Q122868"
            qid_dir.mkdir(parents=True)
            file_one = qid_dir / "111.mp3"
            file_two = qid_dir / "222.mp3"
            file_three = qid_dir / "333.mp3"
            file_one.write_bytes(b"stub-audio-one")
            file_two.write_bytes(b"stub-audio-two")
            file_three.write_bytes(b"stub-audio-three")

            output_dir = root / "embeddings"
            run_dir = output_dir / "birdnet" / "birdnet-resume-test" / "07040000"
            run_dir.mkdir(parents=True)

            manifest_rows = [
                {
                    "audio_id": "Q122868_111_w0000",
                    "qid": "Q122868",
                    "source_path": str(file_one),
                    "relative_path": "Q122868/111.mp3",
                    "window_index": "0",
                    "window_start_seconds": "0.000000",
                    "window_end_seconds": "3.000000",
                    "window_seconds": "3.000000",
                    "file_type": "mp3",
                    "sample_rate": "48000",
                    "num_samples": "144000",
                    "duration_seconds": "3.000000",
                    "embedding_index": "0",
                },
                {
                    "audio_id": "Q122868_111_w0001",
                    "qid": "Q122868",
                    "source_path": str(file_one),
                    "relative_path": "Q122868/111.mp3",
                    "window_index": "1",
                    "window_start_seconds": "3.000000",
                    "window_end_seconds": "6.000000",
                    "window_seconds": "3.000000",
                    "file_type": "mp3",
                    "sample_rate": "48000",
                    "num_samples": "144000",
                    "duration_seconds": "3.000000",
                    "embedding_index": "1",
                },
                {
                    "audio_id": "Q122868_222_w0000",
                    "qid": "Q122868",
                    "source_path": str(file_two),
                    "relative_path": "Q122868/222.mp3",
                    "window_index": "0",
                    "window_start_seconds": "0.000000",
                    "window_end_seconds": "3.000000",
                    "window_seconds": "3.000000",
                    "file_type": "mp3",
                    "sample_rate": "48000",
                    "num_samples": "144000",
                    "duration_seconds": "3.000000",
                    "embedding_index": "2",
                },
            ]
            with (run_dir / "audio_manifest.partial.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, delimiter="	", fieldnames=audio_embeddings.MANIFEST_COLUMNS + ["embedding_index"])
                writer.writeheader()
                writer.writerows(manifest_rows)

            partial_embeddings = np.asarray(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                ],
                dtype=np.float32,
            )
            np.save(run_dir / "embeddings.partial.npy", partial_embeddings)
            (run_dir / "audio_ids.partial.json").write_text(
                json.dumps([row["audio_id"] for row in manifest_rows]), encoding="utf-8"
            )
            (run_dir / "qids.partial.json").write_text(
                json.dumps([row["qid"] for row in manifest_rows]), encoding="utf-8"
            )
            metadata = {
                "backend": "birdnet",
                "model_name": "birdnet-resume-test",
                "input_dir": str(input_dir),
                "max_seconds": 30.0,
                "target_sample_rate": 48000,
                "backend_window_seconds": 3.0,
                "backend_overlap_seconds": 0.0,
                "file_extension_whitelist": ["mp3"],
            }
            (run_dir / "metadata.partial.json").write_text(json.dumps(metadata), encoding="utf-8")

            structured_dtype = np.dtype(
                [
                    ("input", "U256"),
                    ("start_time", np.float32),
                    ("end_time", np.float32),
                    ("embedding", np.float32, (3,)),
                ]
            )

            class FakeBirdNETEncoder:
                device = "cuda"
                sample_rate = 48000
                model_type = "acoustic"
                model_version = "2.4"
                backend = "pb"

                def __init__(self):
                    self.calls = []

                def encode_files(self, paths, *, max_audio_duration_min=None):
                    self.calls.append(([str(path) for path in paths], max_audio_duration_min))
                    if paths != [file_three]:
                        raise AssertionError(f"unexpected resumed paths: {paths}")
                    rows = np.zeros(1, dtype=structured_dtype)
                    rows[0] = (str(file_three), 0.0, 3.0, np.asarray([10.0, 11.0, 12.0], dtype=np.float32))
                    return rows

            fake_encoder = FakeBirdNETEncoder()
            result = audio_embeddings.build_audio_embeddings(
                input_dir=input_dir,
                output_dir=output_dir,
                backend="birdnet",
                model_name="birdnet-resume-test",
                device="cuda",
                batch_size=32,
                max_seconds=30.0,
                target_sample_rate=48000,
                extensions=("mp3",),
                cache_dir=None,
                encoder=fake_encoder,
                resume_existing=True,
            )

            self.assertEqual(fake_encoder.calls, [([str(file_three)], None)])
            self.assertEqual(result["summary"]["reused_item_count"], 3)
            self.assertEqual(result["summary"]["new_item_count"], 1)
            self.assertEqual(result["summary"]["resume_source_run_count"], 1)
            self.assertTrue(result["summary"]["resume_existing"])

            run_dir = Path(result["summary"]["run_dir"])
            audio_ids = json.loads((run_dir / "audio_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(
                audio_ids,
                ["Q122868_111_w0000", "Q122868_111_w0001", "Q122868_222_w0000", "Q122868_333_w0000"],
            )
            embeddings = np.load(run_dir / "embeddings.npy")
            self.assertEqual(embeddings.shape, (4, 3))

    def test_perch_encoder_uses_legacy_official_model_class(self) -> None:
        class FakeBMZ:
            class Perch:
                def __init__(self):
                    self.created = True

        with mock.patch.dict("sys.modules", {"bioacoustics_model_zoo": FakeBMZ}):
            encoder = audio_embeddings.PerchAudioEncoder(device="cpu")

        self.assertEqual(encoder.device, "cpu")
        self.assertTrue(encoder.model.created)
        self.assertEqual(encoder.model_type, "Perch")

    def test_perch_backend_uses_official_clip_dataframe_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "trimmed_xeno_data"
            qid_dir = input_dir / "Q122868"
            qid_dir.mkdir(parents=True)
            audio_path = qid_dir / "333.mp3"
            audio_path.write_bytes(b"stub-audio-three")

            class FakePerchEncoder:
                sample_rate = audio_embeddings.DEFAULT_PERCH_SAMPLE_RATE
                model_type = "Perch"
                model_version = ""
                backend = "bioacoustics-model-zoo"

                def __init__(self):
                    self.calls = []

                def embed_clips(self, samples_df, *, batch_size):
                    self.calls.append((samples_df.copy(), batch_size))
                    if list(samples_df.index.names) != ["file", "start_time", "end_time"]:
                        raise AssertionError(f"unexpected index names: {list(samples_df.index.names)}")
                    rows = []
                    for index, clip in enumerate(samples_df.index):
                        rows.append(
                            np.array(
                                [float(index), float(clip[1]), float(clip[2])],
                                dtype=np.float32,
                            )
                        )
                    return np.vstack(rows)

            fake_encoder = FakePerchEncoder()
            output_dir = root / "embeddings"
            with mock.patch.object(audio_embeddings, "_probe_audio_duration_seconds", return_value=7.0):
                result = audio_embeddings.build_audio_embeddings(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    backend="perch",
                    model_name="perch-unit-test",
                    device="cpu",
                    batch_size=2,
                    max_seconds=10.0,
                    target_sample_rate=audio_embeddings.DEFAULT_PERCH_SAMPLE_RATE,
                    extensions=("mp3",),
                    cache_dir=None,
                    encoder=fake_encoder,
                )

            run_dir = Path(result["summary"]["run_dir"])
            embeddings = np.load(run_dir / "embeddings.npy")
            self.assertEqual(embeddings.shape, (2, 3))

            audio_ids = json.loads((run_dir / "audio_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(audio_ids, ["Q122868_333_w0000", "Q122868_333_w0001"])

            manifest_rows = result["manifest_rows"]
            self.assertEqual(manifest_rows[0]["window_index"], "0")
            self.assertEqual(manifest_rows[1]["window_index"], "1")
            self.assertEqual(manifest_rows[0]["window_seconds"], "5.000000")
            self.assertEqual(manifest_rows[1]["window_seconds"], "2.000000")
            self.assertEqual(manifest_rows[0]["sample_rate"], str(audio_embeddings.DEFAULT_PERCH_SAMPLE_RATE))
            self.assertEqual(result["summary"]["item_count"], 2)
            self.assertEqual(result["summary"]["failed_count"], 0)
            self.assertEqual(result["store"].metadata["decoder"], "perch_clip_dataframe_api")
            self.assertEqual(len(fake_encoder.calls), 1)
            samples_df, used_batch_size = fake_encoder.calls[0]
            self.assertEqual(used_batch_size, 2)
            self.assertEqual(
                list(samples_df.index),
                [(str(audio_path), 0.0, 5.0), (str(audio_path), 5.0, 7.0)],
            )


if __name__ == "__main__":
    unittest.main()
