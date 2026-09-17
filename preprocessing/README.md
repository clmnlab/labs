# fMRIPrep pipeline

This pipeline uses the original 3T sagittal MPRAGE for anatomical processing and
FreeSurfer, and the 7T AP/PA acquisitions for resting-state fMRI preprocessing.
Raw data are never modified. Generated data are written under
`/mnt/ext4/KBSI/preproc`.

## Prerequisites

- Docker access (`docker info` must succeed)
- `dcm2niix` available on `PATH`
- The server FreeSurfer license at `/usr/local/freesurfer/7.4.1/license.txt`, or
  another license referenced by the `FS_LICENSE` environment variable

## Run

```bash
cd /home/sungshinkim/projects/rsfmri

# Create the BIDS dataset.
./prepare_bids.py

# Validate before preprocessing.
./validate_bids.sh

# Pilot one participant first.
./run_fmriprep.sh PHD01

# After checking derivatives/sub-PHD01.html, run the remaining participants.
./run_fmriprep.sh PHD02 PHD03 PHD07
```

The full four-participant command is `./run_fmriprep.sh` with no arguments.
Control resources with `NTHREADS`, `OMP_NTHREADS`, and `MEM_MB`, for example:

```bash
NTHREADS=12 OMP_NTHREADS=4 MEM_MB=48000 ./run_fmriprep.sh PHD01
```

## Add a new participant

After matching 3T and 7T source folders for a new participant (for example,
`PHD08`) have been placed under the raw-data directory, add only that
participant to the existing BIDS dataset:

```bash
./prepare_bids.py --participant-label PHD08
./validate_bids.sh
./run_fmriprep.sh PHD08
```

The script discovers the `SAG/DICOM` or `MPRAGE_SAG/DICOM` 3T layout
automatically. It preserves all existing BIDS participants. Do not pass
`--overwrite` for a new participant; that option is only for intentionally
rebuilding a participant already present in BIDS.

## Metadata decisions

- AP BOLD: `PhaseEncodingDirection = j`
- PA reverse-PE EPI: `PhaseEncodingDirection = j-`
- `TotalReadoutTime = 1 / 37.2 = 0.02688172 s`, from the Philips protocol's
  phase-encoding bandwidth
- Each PA EPI sidecar uses `IntendedFor` to link explicitly to all three BOLD
  runs in the same session. `B0FieldIdentifier`/`B0FieldSource` are omitted so
  SDCFlows uses one fieldmap-discovery mechanism and does not register the same
  estimator more than once.
- No `SliceTiming` is supplied because it is unavailable in the source data;
  fMRIPrep will therefore skip slice-timing correction.
- The same 3T T1w is stored under both pre/post sessions. This makes the shared
  anatomical input visible to fMRIPrep 25.2.5 when it filters a multi-session
  participant by session. FreeSurfer still uses one participant ID.

## Output layout

```text
/mnt/ext4/KBSI/preproc/
├── bids/          # generated BIDS input
├── derivatives/   # fMRIPrep and FreeSurfer outputs
└── work/          # temporary fMRIPrep working files
```

The source data remain read-only at `/mnt/ext4/antoine/PHD/raw_data`.
