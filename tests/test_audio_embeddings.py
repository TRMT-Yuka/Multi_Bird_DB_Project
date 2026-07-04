from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from multi_bird_db import audio_embeddings


class AudioEmbeddingTests(unittest.TestCase):
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

    def test_birdnet_encoder_extracts_one_row_per_input_from_encoding_result(self) -> None:
        class FakeWaveform:
            def __init__(self, values):
                self._values = np.asarray(values, dtype=np.float32)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._values

        class FakeEncodingResult:
            def __init__(self):
                self.embeddings = np.asarray(
                    [
                        [[1.0, 2.0], [9.0, 9.0]],
                        [[3.0, 4.0], [5.0, 6.0]],
                    ],
                    dtype=np.float32,
                )
                self.embeddings_masked = np.asarray(
                    [
                        [[False, False], [True, True]],
                        [[False, False], [False, False]],
                    ],
                    dtype=bool,
                )

        class FakeBirdNETModel:
            def __init__(self):
                self.last_device = None
                self.last_batch_size = None

            def encode_arrays(self, items, *, device, batch_size, **kwargs):
                self.last_device = device
                self.last_batch_size = batch_size
                return FakeEncodingResult()

        fake_model = FakeBirdNETModel()
        with mock.patch.object(audio_embeddings, "birdnet_gpu_available", return_value=True):
            encoder = audio_embeddings.BirdNETAudioEncoder(backend="auto", device="cuda", model=fake_model)
        matrix = encoder.encode_batch([FakeWaveform([0.0, 1.0]), FakeWaveform([2.0, 3.0])])
        np.testing.assert_allclose(matrix, np.asarray([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32))
        self.assertEqual(fake_model.last_device, "GPU:0")
        self.assertEqual(fake_model.last_batch_size, 1)

    def test_birdnet_gpu_path_forces_spawn_and_single_worker(self) -> None:
        class FakeBirdNETModel:
            def __init__(self):
                self.last_kwargs = None

            def encode_arrays(self, items, **kwargs):
                self.last_kwargs = kwargs
                return np.asarray([[1.0, 2.0]], dtype=np.float32)

        fake_model = FakeBirdNETModel()
        with (
            mock.patch.object(audio_embeddings, "birdnet_gpu_available", return_value=True),
            mock.patch.object(audio_embeddings.mp, "get_start_method", return_value=None),
            mock.patch.object(audio_embeddings.mp, "set_start_method") as mock_set_start_method,
        ):
            encoder = audio_embeddings.BirdNETAudioEncoder(backend="auto", device="cuda", model=fake_model)
            matrix = encoder.encode_batch([np.asarray([0.0, 1.0], dtype=np.float32)])

        np.testing.assert_allclose(matrix, np.asarray([[1.0, 2.0]], dtype=np.float32))
        mock_set_start_method.assert_called_once_with("spawn", force=True)
        assert fake_model.last_kwargs is not None
        self.assertEqual(fake_model.last_kwargs["device"], "GPU:0")
        self.assertEqual(fake_model.last_kwargs["batch_size"], 1)
        self.assertEqual(fake_model.last_kwargs["n_workers"], 1)
        self.assertEqual(fake_model.last_kwargs["n_producers"], 1)

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

                def encode_batch(self, waveforms):
                    rows = []
                    for index, waveform in enumerate(waveforms):
                        rows.append(
                            np.array(
                                [
                                    float(index),
                                    float(waveform.shape[0]),
                                    float(np.mean(waveform)),
                                ],
                                dtype=np.float32,
                            )
                        )
                    return np.vstack(rows)

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
                audio_loader=fake_loader,
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

    def test_birdnet_2_backend_uses_official_file_api(self) -> None:
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

            class FakeBirdNET2Encoder:
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

            fake_encoder = FakeBirdNET2Encoder()
            output_dir = root / "embeddings"
            result = audio_embeddings.build_audio_embeddings(
                input_dir=input_dir,
                output_dir=output_dir,
                backend="birdnet_2",
                model_name="birdnet2-unit-test",
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

    def test_perch_backend_uses_five_second_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "trimmed_xeno_data"
            qid_dir = input_dir / "Q122868"
            qid_dir.mkdir(parents=True)
            (qid_dir / "333.mp3").write_bytes(b"stub-audio-three")

            def fake_loader(path: Path) -> tuple[np.ndarray, int]:
                self.assertEqual(path.name, "333.mp3")
                return np.linspace(-1.0, 1.0, 7 * 22050, dtype=np.float32), 22050

            class FakePerchEncoder:
                sample_rate = 22050
                model_type = "Perch2"
                model_version = ""
                backend = "bioacoustics-model-zoo"

                def encode_batch(self, waveforms):
                    rows = []
                    for index, waveform in enumerate(waveforms):
                        rows.append(
                            np.array(
                                [
                                    float(index),
                                    float(waveform.shape[0]),
                                    float(np.mean(waveform)),
                                    float(waveform[-1]),
                                ],
                                dtype=np.float32,
                            )
                        )
                    return np.vstack(rows)

            output_dir = root / "embeddings"
            result = audio_embeddings.build_audio_embeddings(
                input_dir=input_dir,
                output_dir=output_dir,
                backend="perch",
                model_name="perch-unit-test",
                device="cpu",
                batch_size=2,
                max_seconds=10.0,
                target_sample_rate=22050,
                extensions=("mp3",),
                cache_dir=None,
                audio_loader=fake_loader,
                encoder=FakePerchEncoder(),
            )

            run_dir = Path(result["summary"]["run_dir"])
            embeddings = np.load(run_dir / "embeddings.npy")
            self.assertEqual(embeddings.shape, (2, 4))

            audio_ids = json.loads((run_dir / "audio_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(audio_ids, ["Q122868_333_w0000", "Q122868_333_w0001"])

            manifest_rows = result["manifest_rows"]
            self.assertEqual(manifest_rows[0]["window_index"], "0")
            self.assertEqual(manifest_rows[1]["window_index"], "1")
            self.assertEqual(manifest_rows[0]["window_seconds"], "5.000000")
            self.assertEqual(manifest_rows[1]["window_seconds"], "5.000000")
            self.assertEqual(result["summary"]["item_count"], 2)
            self.assertEqual(result["summary"]["failed_count"], 0)


if __name__ == "__main__":
    unittest.main()
