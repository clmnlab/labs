#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bids_dir="${BIDS_DIR:-/mnt/ext4/KBSI/preproc/bids}"
derivatives_dir="${DERIVATIVES_DIR:-/mnt/ext4/KBSI/preproc/derivatives}"
work_dir="${WORK_DIR:-/mnt/ext4/KBSI/preproc/work}"
fs_license="${FS_LICENSE:-/usr/local/freesurfer/7.4.1/license.txt}"
image="nipreps/fmriprep:25.2.5"

if [[ ! -f "${bids_dir}/dataset_description.json" ]]; then
  echo "BIDS dataset not found. Run ./prepare_bids.py first." >&2
  exit 1
fi
if [[ ! -s "${fs_license}" ]]; then
  echo "FreeSurfer license not found: ${fs_license}" >&2
  echo "Place it there or set FS_LICENSE=/absolute/path/license.txt" >&2
  exit 1
fi
docker info >/dev/null
mkdir -p "${derivatives_dir}" "${work_dir}"

participant_args=()
if (($#)); then
  participant_args=(--participant-label "$@")
fi

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${bids_dir}:/data:ro" \
  -v "${derivatives_dir}:/out" \
  -v "${work_dir}:/work" \
  -v "${fs_license}:/opt/freesurfer/license.txt:ro" \
  "${image}" \
  /data /out participant \
  --work-dir /work \
  --fs-license-file /opt/freesurfer/license.txt \
  --output-spaces MNI152NLin2009cAsym:res-2 T1w fsnative \
  --nthreads "${NTHREADS:-8}" \
  --omp-nthreads "${OMP_NTHREADS:-4}" \
  --mem-mb "${MEM_MB:-30000}" \
  --fd-spike-threshold 0.5 \
  --dvars-spike-threshold 1.5 \
  "${participant_args[@]}"
