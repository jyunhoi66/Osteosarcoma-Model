#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radiomics feature extraction from 3D medical images and ROI masks.

This script resamples each image-mask pair to a fixed matrix size and extracts
handcrafted radiomics features using PyRadiomics. It is designed for reproducible
batch processing and can be directly released with a scientific manuscript.

Example
-------
python extract_radiomics_features.py \
    --image-dir /path/to/img \
    --mask-dir /path/to/mask \
    --output-csv /path/to/radiomics_features.csv \
    --target-size 96 96 96

Input requirements
------------------
1. Images and masks should be stored as NIfTI files with the suffix ".nii.gz".
2. Image and mask filenames should share the same patient ID, for example:
   image: patient001.nii.gz
   mask : patient001.nii.gz
3. Mask voxels should contain the ROI label. The default ROI label is 1.

Dependencies
------------
SimpleITK
numpy
pandas
pyradiomics
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


NumericValue = Union[int, float, np.integer, np.floating]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract 3D radiomics features from image-mask pairs."
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing input image files in .nii.gz format.",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        required=True,
        help="Directory containing ROI mask files in .nii.gz format.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to save the extracted radiomics features as a CSV file.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=3,
        default=(96, 96, 96),
        metavar=("X", "Y", "Z"),
        help="Target matrix size after resampling. Default: 96 96 96.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        help="ROI label value in the mask. Default: 1.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=25,
        help="Gray-level bin width used by PyRadiomics. Default: 25.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Enable intensity normalization in PyRadiomics. Default: enabled.",
    )
    parser.add_argument(
        "--disable-normalize",
        action="store_false",
        dest="normalize",
        help="Disable intensity normalization in PyRadiomics.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )

    return parser.parse_args()


def setup_logging() -> None:
    """Configure logging format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_patient_id(filename: str) -> str:
    """
    Extract patient ID from a NIfTI filename.

    Examples
    --------
    patient001.nii.gz -> patient001
    patient001.nii    -> patient001
    """
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def resample_to_size(
    image: sitk.Image,
    mask: sitk.Image,
    target_size: Tuple[int, int, int] = (96, 96, 96),
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Resample image and mask to a fixed matrix size.

    The physical field of view is approximately preserved by adjusting the output
    spacing according to the original image size and spacing. B-spline
    interpolation is used for the image, while nearest-neighbor interpolation is
    used for the mask to preserve discrete labels.

    Parameters
    ----------
    image : sitk.Image
        Input 3D medical image.
    mask : sitk.Image
        Input ROI mask.
    target_size : tuple of int
        Target output matrix size in x, y, and z dimensions.

    Returns
    -------
    resampled_image : sitk.Image
        Resampled image.
    resampled_mask : sitk.Image
        Resampled mask.
    """
    original_size = np.array(image.GetSize(), dtype=np.float64)
    original_spacing = np.array(image.GetSpacing(), dtype=np.float64)
    target_size_array = np.array(target_size, dtype=np.float64)

    new_spacing = original_spacing * original_size / target_size_array

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetSize([int(v) for v in target_size])
    resampler.SetOutputSpacing([float(v) for v in new_spacing])

    resampler.SetInterpolator(sitk.sitkBSpline)
    resampled_image = resampler.Execute(image)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_mask = resampler.Execute(mask)

    # Ensure the mask uses an integer type for PyRadiomics.
    resampled_mask = sitk.Cast(resampled_mask, sitk.sitkUInt8)

    return resampled_image, resampled_mask


def build_radiomics_extractor(
    bin_width: float = 25,
    normalize: bool = True,
    label: int = 1,
) -> featureextractor.RadiomicsFeatureExtractor:
    """
    Build a PyRadiomics feature extractor with predefined settings.

    Notes
    -----
    The image and mask are already resampled to a fixed matrix size before
    feature extraction. Therefore, PyRadiomics internal resampling is disabled by
    setting resampledPixelSpacing to None.
    """
    settings = {
        "force2D": False,
        "normalize": normalize,
        "binWidth": bin_width,
        "interpolator": sitk.sitkBSpline,
        "resampledPixelSpacing": None,
        "voxelBased": False,
        "label": label,
        "additionalInfo": True,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # Enable selected image filters.
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("Wavelet")
    extractor.enableImageTypeByName(
        "LoG",
        customArgs={"sigma": [1.0, 3.0, 5.0]},
    )
    extractor.enableImageTypeByName("Square")
    extractor.enableImageTypeByName("SquareRoot")
    extractor.enableImageTypeByName("Logarithm")
    extractor.enableImageTypeByName("Exponential")
    extractor.enableImageTypeByName("Gradient")
    extractor.enableImageTypeByName(
        "LBP2D",
        customArgs={"radius": 2},
    )

    # Enable selected feature classes.
    extractor.disableAllFeatures()
    for feature_class in ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]:
        extractor.enableFeatureClassByName(feature_class)

    return extractor


def is_numeric_feature(value: object) -> bool:
    """Return True if a PyRadiomics output value is a scalar numeric feature."""
    return isinstance(value, (int, float, np.integer, np.floating))


def clean_feature_dict(result: Dict[str, object]) -> Dict[str, float]:
    """
    Keep scalar numeric radiomics features and remove diagnostic metadata.

    PyRadiomics may return diagnostic fields and other non-feature outputs. This
    function keeps only scalar numeric values to make the CSV suitable for
    downstream statistical analysis or machine-learning workflows.
    """
    cleaned = {}

    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue

        if is_numeric_feature(value):
            cleaned[key] = float(value)

    return cleaned


def validate_mask(mask: sitk.Image, label: int = 1) -> None:
    """
    Check whether the target label exists in the mask.

    Raises
    ------
    ValueError
        If the target label is absent.
    """
    mask_array = sitk.GetArrayViewFromImage(mask)
    if not np.any(mask_array == label):
        raise ValueError(f"ROI label {label} was not found in the mask.")


def iter_image_files(image_dir: Path) -> Iterable[Path]:
    """Yield NIfTI image files from the image directory."""
    for file_path in sorted(image_dir.iterdir()):
        if file_path.name.endswith(".nii.gz") or file_path.name.endswith(".nii"):
            yield file_path


def extract_features_for_case(
    image_path: Path,
    mask_path: Path,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    target_size: Tuple[int, int, int],
    label: int,
) -> Dict[str, float]:
    """Read, resample, validate, and extract radiomics features for one case."""
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))

    logging.info(
        "Image spacing: %s | Mask spacing: %s",
        image.GetSpacing(),
        mask.GetSpacing(),
    )

    resampled_image, resampled_mask = resample_to_size(
        image=image,
        mask=mask,
        target_size=target_size,
    )

    validate_mask(resampled_mask, label=label)

    result = extractor.execute(resampled_image, resampled_mask)
    features = clean_feature_dict(result)

    return features


def main() -> None:
    """Main function for batch radiomics feature extraction."""
    setup_logging()
    args = parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_csv = Path(args.output_csv)
    target_size = tuple(args.target_size)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    if output_csv.exists() and not args.force_overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_csv}. "
            "Use --force-overwrite to overwrite it."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    extractor = build_radiomics_extractor(
        bin_width=args.bin_width,
        normalize=args.normalize,
        label=args.label,
    )

    all_features: List[Dict[str, object]] = []
    failed_cases: List[str] = []

    image_files = list(iter_image_files(image_dir))
    logging.info("Found %d image files.", len(image_files))

    for image_path in image_files:
        patient_id = get_patient_id(image_path.name)
        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            logging.warning("Mask not found for %s. Skipped.", patient_id)
            failed_cases.append(patient_id)
            continue

        logging.info("Processing patient: %s", patient_id)

        try:
            features = extract_features_for_case(
                image_path=image_path,
                mask_path=mask_path,
                extractor=extractor,
                target_size=target_size,
                label=args.label,
            )
            features["PatientID"] = patient_id
            all_features.append(features)

        except Exception as exc:
            logging.exception("Failed to process %s: %s", patient_id, exc)
            failed_cases.append(patient_id)

    if not all_features:
        raise RuntimeError("No valid radiomics features were extracted.")

    features_df = pd.DataFrame(all_features)

    # Put PatientID in the first column.
    columns = ["PatientID"] + [col for col in features_df.columns if col != "PatientID"]
    features_df = features_df[columns]

    features_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    logging.info("Feature extraction completed.")
    logging.info("Saved features to: %s", output_csv)
    logging.info("Number of successful cases: %d", len(all_features))
    logging.info("Number of failed or skipped cases: %d", len(failed_cases))

    if failed_cases:
        logging.warning("Failed or skipped patient IDs: %s", ", ".join(failed_cases))


if __name__ == "__main__":
    main()
