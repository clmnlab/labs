#!/usr/bin/env python3
"""Create the PHD BIDS dataset without modifying the source data."""

import argparse
import gzip
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


SESSIONS = ("pre", "post")
TR = 2.5
TE = 0.022
FLIP_ANGLE = 80
# Philips protocol: phase-encoding bandwidth = 37.2 Hz/pixel.
TOTAL_READOUT_TIME = 1.0 / 37.2


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def copy_nifti_gz(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.name.endswith(".nii.gz"):
        shutil.copy2(source, destination)
    else:
        with source.open("rb") as src, gzip.open(destination, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)


def convert_3t_t1(subject: str, raw: Path, output: Path) -> None:
    dicom_dirs = [
        raw / "3T" / subject / "MPRAGE_SAG" / "DICOM",
        raw / "3T" / subject / "SAG" / "DICOM",
    ]
    dicom_dir = next((path for path in dicom_dirs if (path / "IM_0001").is_file()), None)
    if dicom_dir is None:
        tried = ", ".join(str(path / "IM_0001") for path in dicom_dirs)
        raise FileNotFoundError(f"Original sagittal MPRAGE is missing; tried: {tried}")

    dcm2niix = shutil.which("dcm2niix")
    if dcm2niix is None:
        fallback = Path("/usr/local/fsl/bin/dcm2niix")
        if fallback.is_file():
            dcm2niix = str(fallback)
        else:
            raise FileNotFoundError("dcm2niix is not available on PATH or under /usr/local/fsl/bin")

    with tempfile.TemporaryDirectory(prefix=f"dcm2niix_{subject}_") as tmp_name:
        tmp = Path(tmp_name)
        subprocess.run(
            [dcm2niix, "-b", "y", "-ba", "y", "-z", "y", "-f", "%p_%s", "-o", str(tmp), str(dicom_dir)],
            check=True,
        )
        candidates = []
        for sidecar in tmp.glob("*.json"):
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            image_type = " ".join(str(x) for x in metadata.get("ImageType", []))
            description = " ".join(
                str(metadata.get(key, "")) for key in ("SeriesDescription", "ProtocolName")
            )
            nifti = sidecar.with_suffix(".nii.gz")
            if nifti.is_file() and "MPRAGE" in description.upper():
                score = ("ORIGINAL" in image_type.upper(), nifti.stat().st_size)
                candidates.append((score, nifti, metadata))
        if not candidates:
            raise RuntimeError(f"No original MPRAGE conversion found for {subject} in {tmp}")
        _, nifti, metadata = max(candidates, key=lambda item: item[0])
        # fMRIPrep 25.2.5 filters anatomical inputs by session when multiple
        # sessions are processed together. Store the shared 3T scan under each
        # 7T session so it is available to both session queries.
        for session in SESSIONS:
            anat = output / f"sub-{subject}" / f"ses-{session}" / "anat"
            anat.mkdir(parents=True, exist_ok=True)
            stem = f"sub-{subject}_ses-{session}_T1w"
            shutil.copy2(nifti, anat / f"{stem}.nii.gz")
            write_json(anat / f"{stem}.json", metadata)


def add_7t_session(subject: str, session: str, raw: Path, output: Path) -> None:
    source = raw / "7T" / subject / f"{subject}_{session}"
    func = output / f"sub-{subject}" / f"ses-{session}" / "func"
    fmap = output / f"sub-{subject}" / f"ses-{session}" / "fmap"
    func.mkdir(parents=True, exist_ok=True)
    fmap.mkdir(parents=True, exist_ok=True)
    for run in (1, 2, 3):
        stem = f"sub-{subject}_ses-{session}_task-rest_run-{run}_bold"
        copy_nifti_gz(source / f"fMRI_AP{run}.nii", func / f"{stem}.nii.gz")
        write_json(
            func / f"{stem}.json",
            {
                "TaskName": "rest",
                "MagneticFieldStrength": 7,
                "RepetitionTime": TR,
                "EchoTime": TE,
                "FlipAngle": FLIP_ANGLE,
                "PhaseEncodingDirection": "j",
                "TotalReadoutTime": TOTAL_READOUT_TIME,
            },
        )

    stem = f"sub-{subject}_ses-{session}_dir-PA_epi"
    copy_nifti_gz(source / "fMRI_PA.nii", fmap / f"{stem}.nii.gz")
    write_json(
        fmap / f"{stem}.json",
        {
            "MagneticFieldStrength": 7,
            "RepetitionTime": TR,
            "EchoTime": TE,
            "FlipAngle": FLIP_ANGLE,
            "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": TOTAL_READOUT_TIME,
            "IntendedFor": [
                f"ses-{session}/func/sub-{subject}_ses-{session}_task-rest_run-{run}_bold.nii.gz"
                for run in (1, 2, 3)
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("/mnt/ext4/antoine/PHD/raw_data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/ext4/KBSI/preproc/bids"),
    )
    parser.add_argument(
        "--participant-label",
        nargs="+",
        help="Only add these participants (for example: PHD08 PHD09).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    available_3t = {path.name for path in (args.raw / "3T").iterdir() if path.is_dir()}
    available_7t = {path.name for path in (args.raw / "7T").iterdir() if path.is_dir()}
    available = available_3t & available_7t
    if args.participant_label:
        subjects = tuple(label.removeprefix("sub-") for label in args.participant_label)
        missing = sorted(set(subjects) - available)
        if missing:
            raise SystemExit(f"Participants missing matching 3T and 7T source folders: {', '.join(missing)}")
    else:
        subjects = tuple(sorted(available))
    if not subjects:
        raise SystemExit("No participants with matching 3T and 7T source folders were found")

    existing = [subject for subject in subjects if (args.output / f"sub-{subject}").exists()]
    if existing and not args.overwrite:
        labels = ", ".join(existing)
        raise SystemExit(
            f"BIDS participant(s) already exist: {labels}. Select only new participants, "
            "or pass --overwrite to replace those participants."
        )
    args.output.mkdir(parents=True, exist_ok=True)

    write_json(
        args.output / "dataset_description.json",
        {"Name": "PHD 3T structural and 7T resting-state fMRI", "BIDSVersion": "1.10.1", "DatasetType": "raw"},
    )
    all_subjects = {path.name.removeprefix("sub-") for path in args.output.glob("sub-*") if path.is_dir()}
    all_subjects.update(subjects)
    (args.output / "participants.tsv").write_text(
        "participant_id\n" + "".join(f"sub-{s}\n" for s in sorted(all_subjects)), encoding="utf-8"
    )
    (args.output / "README").write_text(
        "3T original sagittal MPRAGE is used as T1w. 7T AP runs are resting-state BOLD; "
        "the 7T PA acquisition is the reverse phase-encoding image. SliceTiming was not "
        "available and was not inferred.\n",
        encoding="utf-8",
    )
    for subject in subjects:
        convert_3t_t1(subject, args.raw, args.output)
        for session in SESSIONS:
            add_7t_session(subject, session, args.raw, args.output)
    print(f"Added BIDS participant(s): {', '.join(subjects)}")
    print(f"BIDS dataset: {args.output}")


if __name__ == "__main__":
    main()
