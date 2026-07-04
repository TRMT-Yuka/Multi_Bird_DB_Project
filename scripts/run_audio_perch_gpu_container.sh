#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

image_tag="${AUDIO_PERCH_GPU_IMAGE_TAG:-multi-bird-db/audio-perch-gpu:local}"
base_image="${AUDIO_PERCH_GPU_BASE_IMAGE:-nvcr.io/nvidia/tensorflow:25.02-tf2-py3}"
dockerfile_path="${AUDIO_PERCH_GPU_DOCKERFILE:-Dockerfile.audio-perch-gpu}"
build_mode="${AUDIO_PERCH_GPU_BUILD:-auto}"

build_image() {
  docker build     --build-arg BASE_IMAGE="${base_image}"     -f "${repo_root}/${dockerfile_path}"     -t "${image_tag}"     "${repo_root}"
}

if [[ "${1:-}" == "--build-only" ]]; then
  build_image
  exit 0
fi

if [[ "${build_mode}" == "1" || "${build_mode}" == "true" || "${build_mode}" == "yes" ]]; then
  build_image
elif [[ "${build_mode}" == "auto" ]]; then
  if ! docker image inspect "${image_tag}" >/dev/null 2>&1; then
    build_image
  fi
fi

if [[ $# -eq 0 ]]; then
  set -- bash
fi

docker_args=(
  --rm
  --gpus all
  --ipc host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp
  -e USER=trmt
  -e LOGNAME=trmt
  -e USERNAME=trmt
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor
  -e PYTHONPATH=src
  -e HF_HOME=/workspace/data/external/models/audio/huggingface
  -v "${repo_root}:/workspace"
  -w /workspace
)

if [[ -t 0 && -t 1 ]]; then
  docker_args=(-it "${docker_args[@]}")
fi

exec docker run "${docker_args[@]}" "${image_tag}" "$@"
