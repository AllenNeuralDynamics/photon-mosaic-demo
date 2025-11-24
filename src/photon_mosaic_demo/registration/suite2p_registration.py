import copy
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime as dt
from functools import partial
from glob import glob
from itertools import product
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import cv2
import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
import suite2p
from aind_data_schema.core.processing import DataProcess
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status
from aind_data_schema_models.process_names import ProcessName
from aind_log_utils.log import setup_logging
from aind_ophys_utils.array_utils import normalize_array
from aind_ophys_utils.summary_images import mean_image
from aind_ophys_utils.video_utils import (
    downsample_array,
    downsample_h5_video,
    encode_video,
)
from aind_qcportal_schema.metric_value import DropdownMetric
from matplotlib import pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont
from ScanImageTiffReader import ScanImageTiffReader
from scipy.ndimage import median_filter
from suite2p.registration.nonrigid import make_blocks
from suite2p.registration.register import register_frames
from suite2p.registration.rigid import (
    shift_frame,
)

from photon_mosaic_demo.core.base_processor import BaseProcessor
from photon_mosaic_demo.registration.model import MotionCorrectionSettings
from photon_mosaic_demo.registration.reference_image import (
    generate_single_plane_reference,
    h5py_to_numpy,
    tiff_to_numpy,
    update_suite2p_args_reference_image,
)

mpl.use("Agg")


def is_S3(file_path: str):
    """Test if a file is in a S3 bucket
    Parameters
    ----------
    file_path : str
        Location of the file.
    """
    return "s3fs" in subprocess.check_output("df " + file_path + "| sed -n '2 p'", shell=True, text=True)


def h5py_byteorder_name(h5py_file: h5py.File, h5py_key: str) -> Tuple[str, str]:
    """Get the byteorder and name of the dataset in the h5py file.

    Parameters
    ----------
    h5py_file : h5py.File
        h5py file object
    h5py_key : str
        key to the dataset

    Returns
    -------
    str
        byteorder of the dataset
    str
        name of the dataset
    """
    with h5py.File(h5py_file, "r") as f:
        byteorder = f[h5py_key].dtype.byteorder
        name = f[h5py_key].dtype.name
    return byteorder, name


def tiff_byteorder_name(tiff_file: Path) -> Tuple[str, str]:
    """Get the byteorder and name of the dataset in the tiff file.

    Parameters
    ----------
    tiff_file : Path
        Location of the tiff file

    Returns
    -------
    str
        byteorder of the dataset
    str
        name of the dataset
    """
    with ScanImageTiffReader(tiff_file) as reader:
        byteorder = reader.data().dtype.byteorder
        name = reader.data().dtype.name
    return byteorder, name


def combine_images_with_individual_titles(
    image1_path: Path, image2_path: Path, output_path: Path, title1: str, title2: str
) -> None:
    """Combine two images side-by-side with padding and titles above each image.

    Parameters
    ----------
    image1_path : Path
        Path to the first image.
    image2_path : Path
        Path to the second image.
    output_path : Path
        Path to save the combined image.
    title1 : str
         Title text for the first image.
    title2 - str
        Title text for the second image.

    Returns
    -------
    None
    """
    # Open both images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Ensure both images have the same height
    max_height = max(img1.height, img2.height)
    img1 = img1.resize((img1.width, max_height), Image.Resampling.LANCZOS)
    img2 = img2.resize((img2.width, max_height), Image.Resampling.LANCZOS)

    # Set padding and title height
    padding = 20
    title_height = 50  # Space for the titles

    # Calculate dimensions of the combined image
    combined_width = img1.width + img2.width + padding * 3  # Padding between and around images
    combined_height = max_height + padding * 2 + title_height

    # Create a new blank image with padding and room for the titles
    combined_image = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))

    # Draw the titles
    draw = ImageDraw.Draw(combined_image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)  # You can replace with a path to your desired font
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font; may not match expected size

    # Title 1: Above the second image
    bbox1 = draw.textbbox((0, 0), title1, font=font)
    text_width1 = bbox1[2] - bbox1[0]
    text_y1 = padding
    text_x1 = padding + (img1.width - text_width1) // 2
    draw.text((text_x1, text_y1), title1, fill="black", font=font)

    # Title 2: Above the first image
    bbox2 = draw.textbbox((0, 0), title2, font=font)
    text_width2 = bbox2[2] - bbox2[0]
    text_x2 = padding * 2 + img2.width + (img2.width - text_width2) // 2
    text_y2 = padding
    draw.text((text_x2, text_y2), title2, fill="black", font=font)

    # Paste images into the new image
    img1_x = padding
    img2_x = img1_x + img1.width + padding
    img_y = padding + title_height
    combined_image.paste(img1, (img1_x, img_y))
    combined_image.paste(img2, (img2_x, img_y))

    # Save the result
    combined_image.save(output_path)


def serialize_registration_summary_qcmetric(output_dir: Path) -> None:
    """Serialize the registration summary QCMetric

    Parameters
    ----------
    output_dir : Path
        Output directory containing the registration summary PNG

    QCMetric is named 'registration_summary_metric.json' and is
    saved to the same directory as *_registration_summary.png.
    Ex: '/results/<unique_id>/motion_correction/'
    """

    file_path = next(output_dir.rglob("*_registration_summary.png"))

    # Remove '/results' from file_path
    reference_filepath = Path(*file_path.parts[2:])
    unique_id = reference_filepath.parts[0]

    metric = QCMetric(
        name=f"{unique_id} Registration Summary",
        description="Review the registration summary plot to ensure that the motion correction is accurate and sufficient.",
        reference=str(reference_filepath),
        status_history=[QCStatus(evaluator="Pending review", timestamp=dt.now(), status=Status.PENDING)],
        value=DropdownMetric(
            value=[],
            options=[
                "Motion correction successful",
                "No motion correction applied",
                "Motion correction failed",
                "Motion correction partially successful",
            ],
            status=[Status.PASS, Status.FAIL, Status.FAIL, Status.FAIL],
        ),
    )

    with open(Path(file_path.parent) / f"{unique_id}_registration_summary_metric.json", "w") as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)


def serialize_fov_quality_qcmetric(output_dir: Path) -> None:
    """Serialize the FOV Quality QCMetric

    Parameters
    ----------
    output_dir : Path
        Output directory containing the projection PNGs

    QCMetric is named 'fov_quality_metric.json' and is
    saved to the same directory as *_maximum_projection.png.
    Ex: '/results/<unique_id>/motion_correction/'
    """

    avg_projection_file_path = next(output_dir.rglob("*_average_projection.png"))
    max_projection_file_path = next(output_dir.rglob("*_maximum_projection.png"))

    file_path = Path(str(max_projection_file_path).replace("maximum", "combined"))

    combine_images_with_individual_titles(
        avg_projection_file_path,
        max_projection_file_path,
        file_path,
        title1="Average Projection",
        title2="Maximum Projection",
    )

    # Remove /results from file_path
    reference_filepath = Path(*file_path.parts[2:])
    unique_id = reference_filepath.parts[0]

    metric = QCMetric(
        name=f"{unique_id} FOV Quality",
        description="Review the avg. and max. projections to ensure that the FOV quality is sufficient.",
        reference=str(reference_filepath),
        status_history=[QCStatus(evaluator="Pending review", timestamp=dt.now(), status=Status.PENDING)],
        value=DropdownMetric(
            value=["Quality is sufficient"],
            options=[
                "Quality is sufficient",
                "Timeseries shuffled between planes",
                "Field of view associated with incorrect area and/or depth",
                "Paired plane cross talk: Extreme",
                "Paired plane cross-talk: Moderate",
            ],
            status=[Status.PASS, Status.FAIL, Status.FAIL, Status.FAIL, Status.FAIL],
        ),
    )

    with open(Path(file_path.parent) / f"{unique_id}_fov_quality_metric.json", "w") as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)


def compute_residual_optical_flow(
    reg_pc: Union[h5py.Dataset, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the residual optical flow from the registration principal
    components.

    Parameters
    ----------
    reg_pc : Union[h5py.Dataset, np.ndarray]
        Registration principal components.

    Returns
    -------
    residual optical flow : np.ndarray
    average and max shifts : np.ndarray
    """

    regPC = reg_pc
    flows = np.zeros(regPC.shape[1:] + (2,), np.float32)
    for i in range(len(flows)):
        pclow, pchigh = regPC[:, i]
        flows[i] = cv2.calcOpticalFlowFarneback(
            pclow,
            pchigh,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=100,
            iterations=15,
            poly_n=5,
            poly_sigma=1.2 / 5,
            flags=0,
        )
    flows_norm = np.sqrt(np.sum(flows**2, -1))
    farnebackDX = np.transpose([flows_norm.mean((1, 2)), flows_norm.max((1, 2))])
    return flows, farnebackDX


def compute_crispness(
    mov_raw: Union[h5py.Dataset, np.ndarray], mov_corr: Union[h5py.Dataset, np.ndarray]
) -> List[float]:
    """Compute the crispness of the raw and corrected movie.

    Parameters
    ----------
    mov_raw : Union[h5py.Dataset, np.ndarray]
        Raw movie data.
    mov_corr : Union[h5py.Dataset, np.ndarray]
        Corrected movie data.

    Returns
    -------
    crispness of mean image : List[float]
    """
    return [np.sqrt(np.sum(np.array(np.gradient(mean_image(m))) ** 2)) for m in (mov_raw, mov_corr)]


def load_representative_sub_frames(
    h5py_name,
    h5py_key,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
    n_batches: int = 20,
    batch_size: int = 500,
):
    """Load a subset of frames spanning the full movie.

    Parameters
    ----------
    h5py_name : str
        Path to the h5 file to load frames from.
    h5py_key : str
        Name of the h5 dataset containing the movie.
    trim_frames_start : int, optional
        Number of frames to disregard from the start of the movie. Default 0.
    trim_frames_start : int, optional
        Number of frames to disregard from the end of the movie. Default 0.
    n_batches : int
        Number of batches to load. Total returned size is
        n_batches * batch_size.
    batch_size : int, optional
        Number of frames to process at once. Total returned size is
        n_batches * batch_size.

    Returns
    -------
    """
    output_frames = []
    frame_fracts = np.arange(0, 1, 1 / n_batches)
    with h5py.File(h5py_name, "r") as h5_file:
        dataset = h5_file[h5py_key]
        total_frames = dataset.shape[0] - trim_frames_start - trim_frames_end
        if total_frames < n_batches * batch_size:
            return dataset[:]
        for percent_start in frame_fracts:
            frame_start = int(percent_start * total_frames + trim_frames_start)
            output_frames.append(dataset[frame_start : frame_start + batch_size])
    return np.concatenate(output_frames)


def create_ave_image(
    ref_image: np.ndarray,
    input_frames: np.ndarray,
    suite2p_args: dict,
    batch_size: int = 500,
) -> dict:
    """Run suite2p image motion correction over a full movie.

    Parameters
    ----------
    ref_image : numpy.ndarray, (N, M)
        Reference image to correlate with movie frames.
    input_frames : numpy.ndarray, (L, N, M)
        Frames to motion correct and compute average image/acutance of.
    suite2p_args : dict
        Dictionary of suite2p args containing:

        ``"h5py"``
            HDF5 file containing to the movie to motion correct.
        ``"h5py_key"``
            Name of the dataset where the movie to be motion corrected is
            stored.
        ``"maxregshift"``
            Maximum shift allowed as a fraction of the image dimensions.
        ``"smooth_sigma"``
            Spatial Gaussian smoothing parameter used by suite2p to smooth
            frames before correlation. (float).
        ``"smooth_sigma_time"``
            Time Gaussian smoothing of frames to apply before correlation.
            (float).
    batch_size : int, optional
        Number of frames to process at once.

    Returns
    -------
    ave_image_dict : dict
        A dict containing the average image and motion border values:

        ``ave_image``
            Image created with the settings yielding the highest image acutance
            (numpy.ndarray, (N, M))
        ``min_y``
            Minimum y allowed value in image array. Below this is motion
            border.
        ``max_y``
            Maximum y allowed value in image array. Above this is motion
            border.
        ``min_x``
            Minimum x allowed value in image array. Below this is motion
            border.
        ``max_x``
            Maximum x allowed value in image array. Above this is motion
            border.
    """
    ave_frame = np.zeros((ref_image.shape[0], ref_image.shape[1]))
    min_y = 0
    max_y = 0
    min_x = 0
    max_x = 0
    tot_frames = input_frames.shape[0]
    add_modify_required_parameters(suite2p_args)
    for start_idx in np.arange(0, tot_frames, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > tot_frames:
            end_idx = tot_frames
        frames = input_frames[start_idx:end_idx]
        frames, dy, dx, _, _, _, _ = register_frames(refAndMasks=ref_image, frames=frames, ops=suite2p_args)
        min_y = min(min_y, dy.min())
        max_y = max(max_y, dy.max())
        min_x = min(min_x, dx.min())
        max_x = max(max_x, dx.max())
        ave_frame += frames.sum(axis=0) / tot_frames

    return {
        "ave_image": ave_frame,
        "min_y": int(np.fabs(min_y)),
        "max_y": int(max_y),
        "min_x": int(np.fabs(min_x)),
        "max_x": int(max_x),
    }


def add_modify_required_parameters(suite2p_args: dict):
    """Check that minimum parameters needed by suite2p registration are
    available. If not add them to the suite2p_args dict.

    Additionally, make sure that nonrigid is set to false as are gridsearch
    of parameters above is not setup to use nonrigid.

    Parameters
    ----------
    suite2p_args : dict
        Suite2p ops dictionary with potentially missing values.
    """
    if suite2p_args.get("1Preg") is None:
        suite2p_args["1Preg"] = False
    if suite2p_args.get("bidiphase") is None:
        suite2p_args["bidiphase"] = False
    if suite2p_args.get("nonrigid") is None:
        suite2p_args["nonrigid"] = False
    if suite2p_args.get("norm_frames") is None:
        suite2p_args["norm_frames"] = True
    # Don't use nonrigid for parameter search.
    suite2p_args["nonrigid"] = False


def check_and_warn_on_datatype(
    filepath: Union[Path, list],
    logger: Callable,
    filetype: str = "h5",
    h5py_key: str = "",
):
    """Suite2p assumes int16 types throughout code. Check that the input
    data is type int16 else throw a warning.

    Parameters
    ----------
    filepath : Union[Path, list]
        Path to the HDF5 containing the data.
    logger : Callable
        Logger to output logger warning to.
    filetype : str
        Type of file to check. Default is "h5".
    h5py_key : str
        Name of the dataset to check. Default is "".

    """
    if filetype == "h5":
        byteorder, name = h5py_byteorder_name(filepath, h5py_key)
    elif filetype == "tiff":
        byteorder, name = tiff_byteorder_name(filepath)
    else:
        raise ValueError("File type not supported")
    if byteorder == ">":
        logger(
            "Data byteorder is big-endian which may cause issues in "
            "suite2p. This may result in a crash or unexpected "
            "results."
        )
    if name != "int16":
        logger(
            f"Data type is {name} and not int16. Suite2p "
            "assumes int16 data as input and throughout codebase. "
            "Non-int16 data may result in unexpected results or "
            "crashes."
        )


def _mean_of_batch(i, array):
    return array[i : i + 1000].mean(axis=(1, 2))


def find_movie_start_end_empty_frames(
    filepath: Union[str, list[str]],
    h5py_key: str = "",
    n_sigma: float = 5,
    logger: Optional[Callable] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[int, int]:
    """Load a movie from HDF5 and find frames at the start and end of the
    movie that are empty or pure noise and 5 sigma discrepant from the
    average frame.

    If a non-contiguous set of frames is found, the code will return 0 for
    that half of the movie and throw a warning about the quality of the data.

    Parameters
    ----------
    h5py_name : str
        Name of the HDF5 file to load from.
    h5py_key : str
        Name of the dataset to load from the HDF5 file.
    n_sigma : float
        Number of standard deviations beyond which a frame is considered an
        outlier and "empty".
    logger : Optional[Callable]
        Function to print warning messages to.
    n_jobs: Optional[int]
        The number of jobs to run in parallel.

    Returns
    -------
    trim_frames : Tuple[int, int]
        Tuple of the number of frames to cut from the start and end of the
        movie as (n_trim_start, n_trim_end).
    """
    if isinstance(filepath, str):
        array = h5py_to_numpy(filepath, h5py_key)
    else:
        array = tiff_to_numpy(filepath)
    # Find the midpoint of the movie.
    n_frames = array.shape[0]
    midpoint = n_frames // 2
    # We discover empty or extrema frames by comparing the mean of each frames
    # to the mean of the full movie.
    if n_jobs == 1 or n_frames < 2000:
        means = array[:].mean(axis=(1, 2))
    else:
        means = np.concatenate(
            ThreadPool(n_jobs).starmap(
                _mean_of_batch,
                product(range(0, n_frames, 1000), [array]),
            )
        )
    mean_of_frames = means.mean()

    # Compute a robust standard deviation that is not sensitive to the
    # outliers we are attempting to find.
    quart_low, quart_high = np.percentile(means, [25, 75])
    # Convert the inner quartile range to an estimate of the standard deviation
    # 0.6745 is the converting factor between the inner quartile and a
    # traditional standard deviation.
    std_est = (quart_high - quart_low) / (2 * 0.6745)

    # Get the indexes of the frames that are found to be n_sigma deviating.
    start_idxs = np.sort(np.argwhere(means[:midpoint] < mean_of_frames - n_sigma * std_est)).flatten()
    end_idxs = np.sort(np.argwhere(means[midpoint:] < mean_of_frames - n_sigma * std_est)).flatten() + midpoint

    # Get the total number of these frames.
    lowside = len(start_idxs)
    highside = len(end_idxs)

    # Check to make sure that the indexes found were only from the start/end
    # of the movie. If not, throw a warning and reset the number of frames
    # found to zero.
    if not np.array_equal(start_idxs, np.arange(0, lowside, dtype=start_idxs.dtype)):
        lowside = 0
        if logger is not None:
            logger(
                f"{n_sigma} sigma discrepant frames found outside the "
                "beginning of the movie. Please inspect the movie for data "
                "quality. Not trimming frames from the movie beginning."
            )
    if not np.array_equal(
        end_idxs,
        np.arange(n_frames - highside, n_frames, dtype=end_idxs.dtype),
    ):
        highside = 0
        if logger is not None:
            logger(
                f"{n_sigma} sigma discrepant frames found outside the end "
                "of the movie. Please inspect the movie for data quality. "
                "Not trimming frames from the movie end."
            )

    return (lowside, highside)


def reset_frame_shift(
    frames: np.ndarray,
    dy_array: np.ndarray,
    dx_array: np.ndarray,
    trim_frames_start: int,
    trim_frames_end: int,
):
    """Reset the frames of a movie and their shifts.

    Shifts the frame back to its original location and resets the shifts for
    those frames to (0, 0). Frames, dy_array, and dx_array are edited in
    place.

    Parameters
    ----------
    frames : numpy.ndarray, (N, M, K)
        Full movie to reset frames in.
    dy_array : numpy.ndarray, (N,)
        Array of shifts in the y direction for each frame of the movie.
    dx_array : numpy.ndarray, (N,)
        Array of shifts in the x direction for each frame of the movie.
    trim_frames_start : int
        Number of frames at the start of the movie that were identified as
        empty or pure noise.
    trim_frames_end : int
        Number of frames at the end of the movie that were identified as
        empty or pure noise.
    """
    for idx in range(trim_frames_start):
        dy = -dy_array[idx]
        dx = -dx_array[idx]
        frames[idx] = shift_frame(frames[idx], dy, dx)
        dy_array[idx] = 0
        dx_array[idx] = 0

    for idx in range(frames.shape[0] - trim_frames_end, frames.shape[0]):
        dy = -dy_array[idx]
        dx = -dx_array[idx]
        frames[idx] = shift_frame(frames[idx], dy, dx)
        dy_array[idx] = 0
        dx_array[idx] = 0


def projection_process(data: np.ndarray, projection: str = "max") -> np.ndarray:
    """

    Parameters
    ----------
    data: np.ndarray
        nframes x nrows x ncols, uint16
    projection: str
        "max" or "avg"

    Returns
    -------
    proj: np.ndarray
        nrows x ncols, uint8

    """
    if projection == "max":
        proj = np.max(data, axis=0)
    elif projection == "avg":
        proj = np.mean(data, axis=0)
    else:
        raise ValueError('projection can be "max" or "avg" not ' f"{projection}")
    return normalize_array(proj)


def identify_and_clip_outliers(data: np.ndarray, med_filter_size: int, thresh: int) -> Tuple[np.ndarray, np.ndarray]:
    """given data, identify the indices of outliers
    based on median filter detrending, and a threshold

    Parameters
    ----------
    data: np.ndarray
        1D array of samples
    med_filter_size: int
        the number of samples for 'size' in
        scipy.ndimage.filters.median_filter
    thresh: int
        multipled by the noise estimate to establish a threshold, above
        which, samples will be marked as outliers.

    Returns
    -------
    data: np.ndarry
        1D array of samples, clipped to threshold around median-filtered data
    indices: np.ndarray
        the indices where clipping took place

    """
    data_filtered = median_filter(data, med_filter_size, mode="nearest")
    detrended = data - data_filtered
    indices = np.argwhere(np.abs(detrended) > thresh).flatten()
    data[indices] = np.clip(data[indices], data_filtered[indices] - thresh, data_filtered[indices] + thresh)
    return data, indices


def make_output_directory(output_dir: Path, plane: str = "") -> str:
    """Creates the output directory if it does not exist

    Parameters
    ----------
    output_dir: Path
        output directory
    plane: str
        plane number

    Returns
    -------
    output_dir: Path
        output directory
    """
    if not plane:
        output_dir = output_dir / "motion_correction"
    else:
        output_dir = output_dir / plane / "motion_correction"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_data_process(
    metadata: dict,
    raw_movie: Union[str, Path],
    motion_corrected_movie: Union[str, Path],
    output_dir: Union[str, Path],
    unique_id: str,
    start_time: dt,
    end_time: dt,
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    raw_movie: str
        path to raw movies
    motion_corrected_movie: str
        path to motion corrected movies
    """
    if isinstance(raw_movie, Path):
        raw_movie = str(raw_movie)
    if isinstance(motion_corrected_movie, Path):
        motion_corrected_movie = str(motion_corrected_movie)
    data_proc = DataProcess(
        name=ProcessName.VIDEO_MOTION_CORRECTION,
        software_version=os.getenv("VERSION", ""),
        start_date_time=start_time.isoformat(),
        end_date_time=end_time.isoformat(),
        input_location=str(raw_movie),
        output_location=str(motion_corrected_movie),
        code_url=("https://github.com/AllenNeuralDynamics/" "aind-ophys-motion-correction/tree/main/code"),
        parameters=metadata,
    )
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    with open(output_dir / f"{unique_id}_motion_correction_data_process.json", "w") as f:
        json.dump(json.loads(data_proc.model_dump_json()), f, indent=4)


def check_trim_frames(data):
    """Make sure that if the user sets auto_remove_empty_frames
    and timing frames is already requested, raise an error.
    """
    if data["auto_remove_empty_frames"] and (data["trim_frames_start"] > 0 or data["trim_frames_end"] > 0):
        msg = (
            "Requested auto_remove_empty_frames but "
            "trim_frames_start > 0 or trim_frames_end > 0. Please "
            "either request auto_remove_empty_frames or manually set "
            "trim_frames_start/trim_frames_end if number of frames to "
            "trim is known."
        )
        raise ValueError(msg)
    return data


def make_png(max_proj_path: Path, avg_proj_path: Path, summary_df: pd.DataFrame, dst_path: Path):
    """ """
    xo = np.abs(summary_df["x"]).max()
    yo = np.abs(summary_df["y"]).max()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 4)
    mx_ax = fig.add_subplot(gs[0:2, 0:2])
    av_ax = fig.add_subplot(gs[0:2, 2:4])
    xyax = fig.add_subplot(gs[2, :])
    corrax = fig.add_subplot(gs[3, :])

    for ax, im_path in zip([mx_ax, av_ax], [max_proj_path, avg_proj_path]):
        with Image.open(im_path) as im:
            ax.imshow(im, cmap="gray")
            sz = im.size
        ax.axvline(xo, color="r", linestyle="--")
        ax.axvline(sz[0] - xo, color="r", linestyle="--")
        ax.axhline(yo, color="g", linestyle="--")
        ax.axhline(sz[1] - yo, color="g", linestyle="--")
        ax.set_title(f"{im_path.parent}\n{im_path.name}", fontsize=8)

    xyax.plot(summary_df["x"], linewidth=0.5, color="r", label="xoff")
    xyax.axhline(xo, color="r", linestyle="--")
    xyax.axhline(-xo, color="r", linestyle="--")
    xyax.plot(summary_df["y"], color="g", linewidth=0.5, alpha=0.5, label="yoff")
    xyax.axhline(yo, color="g", linestyle="--")
    xyax.axhline(-yo, color="g", linestyle="--")
    xyax.legend(loc=0)
    xyax.set_ylabel("correction offset [pixels]")

    corrax.plot(summary_df["correlation"], color="k", linewidth=0.5, label="corrXY")
    corrax.set_xlabel("frame index")
    corrax.set_ylabel("correlation peak value")
    corrax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(dst_path)

    return dst_path


def make_nonrigid_png(output_path: Path, avg_proj_path: Path, summary_df: pd.DataFrame, dst_path: Path):
    """ """
    nonrigid_y = np.array(list(map(eval, summary_df["nonrigid_y"])), dtype=np.float32)
    nonrigid_x = np.array(list(map(eval, summary_df["nonrigid_x"])), dtype=np.float32)
    nonrigid_corr = np.array(list(map(eval, summary_df["nonrigid_corr"])), dtype=np.float32)
    ops = json.loads(h5py.File(output_path)["metadata"][()].decode())["suite2p_args"]
    with Image.open(avg_proj_path) as im:
        Ly, Lx = im.size
    yblock, xblock = make_blocks(Ly=Ly, Lx=Lx, block_size=ops["block_size"])[:2]
    nblocks = len(xblock)

    fig = plt.figure(figsize=(22, 3 * nblocks))
    gs = fig.add_gridspec(25 * nblocks, 6)
    for i in range(nblocks):
        av_ax = fig.add_subplot(gs[25 * i : 25 * i + 20, 0])
        xyax = fig.add_subplot(gs[25 * i : 25 * i + 10, 1:])
        corrax = fig.add_subplot(gs[25 * i + 10 : 25 * i + 20, 1:])

        with Image.open(avg_proj_path) as im:
            av_ax.imshow(im, cmap="gray")
            sz = im.size
            av_ax.set_ylim(0, sz[0])
            av_ax.set_xlim(0, sz[1])
        for x in xblock[i]:
            av_ax.vlines(x, *yblock[i], color="r", linestyle="--")
        for y in yblock[i]:
            av_ax.hlines(y, *xblock[i], color="g", linestyle="--")

        xyax.plot(nonrigid_x[:, i], linewidth=0.5, color="r", label="xoff")
        xyax.plot(nonrigid_y[:, i], color="g", linewidth=0.5, alpha=0.5, label="yoff")
        if i == 0:
            xyax.legend(loc=0)
        xyax.set_xticks([])
        xyax.set_xlim(0, nonrigid_x.shape[0])
        xyax.set_ylabel("offset [pixels]")

        corrax.plot(nonrigid_corr[:, i], color="k", linewidth=0.5, label="corrXY")
        corrax.set_xlim(0, nonrigid_x.shape[0])
        corrax.set_xlabel("frame index")
        corrax.set_ylabel("correlation")
        if i == 0:
            corrax.legend(loc=0)
    fig.savefig(dst_path, bbox_inches="tight")

    return dst_path


def downsample_normalize(
    movie_path: Path,
    frame_rate: float,
    bin_size: float,
    lower_quantile: float,
    upper_quantile: float,
) -> np.ndarray:
    """reads in a movie (nframes x nrows x ncols), downsamples,
    creates an average projection, and normalizes according to
    quantiles in that projection.

    Parameters
    ----------
    movie_path: Path
        path to an h5 file, containing an (nframes x nrows x ncol) dataset
        named 'data'
    frame_rate: float
        frame rate of the movie specified by 'movie_path'
    bin_size: float
        desired duration in seconds of a downsampled bin, i.e. the reciprocal
        of the desired downsampled frame rate.
    lower_quantile: float
        arg supplied to `np.quantile()` to determine lower cutoff value from
        avg projection for normalization.
    upper_quantile: float
        arg supplied to `np.quantile()` to determine upper cutoff value from
        avg projection for normalization.

    Returns
    -------
    ds: np.ndarray
        a downsampled and normalized array

    Notes
    -----
    This strategy was satisfactory in the labeling app for maintaining
    consistent visibility.

    """
    if isinstance(movie_path, Path):
        ds = downsample_h5_video(movie_path, input_fps=frame_rate, output_fps=1.0 / bin_size)
    else:
        ds = downsample_array(movie_path, input_fps=frame_rate, output_fps=1.0 / bin_size)
    avg_projection = ds.mean(axis=0)
    lower_cutoff, upper_cutoff = np.quantile(avg_projection.flatten(), (lower_quantile, upper_quantile))
    ds = normalize_array(ds, lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff)
    return ds


def flow_png(output_path: Path, dst_path: str, iPC: int = 0):
    with h5py.File(output_path) as f:
        regPC = f["reg_metrics/regPC"]
        tPC = f["reg_metrics/tPC"]
        flows = f["reg_metrics/farnebackROF"]
        flow_ds = np.array([cv2.resize(flows[iPC, :, :, a], dsize=None, fx=0.1, fy=0.1) for a in (0, 1)])
        flow_ds_norm = np.sqrt(np.sum(flow_ds**2, 0))
        # redo Suite2p's PCA-based frame selection
        n_frames, Ly, Lx = f["data"].shape
        nsamp = min(2000 if n_frames < 5000 or Ly > 700 or Lx > 700 else 5000, n_frames)
        inds = np.linspace(0, n_frames - 1, nsamp).astype("int")
        nlowhigh = np.minimum(300, int(n_frames / 2))
        isort = np.argsort(tPC, axis=0)

        for k in (0, 1):
            f, a = plt.subplots(2, figsize=(5, 6))
            a[0].set_position([0.08, 0.92, 0.88, 0.08])
            a[0].hist(
                np.sort(inds[isort[-nlowhigh:, iPC] if k else isort[:nlowhigh, iPC]]),
                50,
            )
            a[0].set_title("averaged frames for " + ("$PC_{high}$" if k else "$PC_{low}$"))
            a[1].set_position([0, 0, 1, 0.9])
            vmin = np.min(regPC[1 if k else 0, iPC])
            vmax = 5 * np.median(regPC[1 if k else 0, iPC]) - 4 * vmin
            a[1].imshow(regPC[1 if k else 0, iPC], cmap="gray", vmin=vmin, vmax=vmax)
            a[1].axis("off")
            plt.savefig(
                dst_path + (f"_PC{iPC}low.png", f"_PC{iPC}high.png")[k],
                format="png",
                dpi=300,
                bbox_inches="tight",
            )
        f, a = plt.subplots(2, figsize=(5, 6))
        a[0].set_position([0.06, 0.95, 0.9, 0.05])
        a[1].set_position([0, 0, 1, 0.9])
        im = a[1].quiver(*flow_ds[:, ::-1], flow_ds_norm[::-1])  # imshow puts origin [0,0] in upper left
        a[1].axis("off")
        plt.colorbar(im, cax=a[0], location="bottom")
        a[0].set_title("residual optical flow")
        plt.savefig(dst_path + f"_PC{iPC}rof.png", format="png", dpi=300, bbox_inches="tight")


def multiplane_motion_correction(data_dir: Path, output_dir: Path, debug: bool = False):
    """Process multiplane data for suite2p parameters

    Parameters
    ----------
    data_dir: Path
        path to h5 file
    output_dir: Path
        output directory
    debug: bool
        run in debug mode
    Returns
    -------
    h5_file: Path
        path to h5 file
    output_dir: Path
        output directory
    frame_rate_hz: float
        frame rate in Hz
    """
    pattern = re.compile(r"^V.*\d+$")
    matching_files = [d for d in data_dir.rglob("*.txt") if pattern.match(d.stem)]
    if len(matching_files) > 0:
        with open(matching_files[0], "r") as f:
            h5_file = f.read()
        h5_file = next(data_dir.rglob(h5_file))
        unique_id = h5_file.stem
    else:
        h5_dir = [i for i in data_dir.rglob("*V*") if i.is_dir()][0]
        unique_id = h5_dir.name
        h5_file = [i for i in h5_dir.glob(f"{h5_dir.name}.h5")][0]
    logging.info("Found raw time series to process %s", h5_file)
    session_fp = next(data_dir.rglob("session.json"), "")
    if not session_fp:
        raise f"Could not locate session.json in {session_fp}"
    with open(session_fp) as f:
        session_data = json.load(f)
    output_dir = make_output_directory(output_dir, unique_id)
    try:
        frame_rate_hz = float(session_data["data_streams"][0]["ophys_fovs"][0]["frame_rate"])
    except KeyError:
        logging.warning("Frame rate not found in session.json, using default")
        frame_rate_hz = 30.0
    if debug:
        logging.info("Running in debug mode....")
        raw_data = h5py.File(h5_file, "r")
        frames_6min = int(360 * float(frame_rate_hz))
        trimmed_data = raw_data["data"][:frames_6min]
        raw_data.close()
        trimmed_fn = Path("../scratch") / f"{unique_id}.h5"
        with h5py.File(trimmed_fn, "w") as f:
            f.create_dataset("data", data=trimmed_data)
        h5_file = trimmed_fn
    return h5_file, output_dir, frame_rate_hz


def singleplane_motion_correction(h5_file: Path, output_dir: Path, session, unique_id: str, debug: bool = False):
    """Process single plane data for suite2p parameters

    Parameters
    ----------
    h5_file: Path
        path to h5 file
    output_dir: Path
        output directory
    session: dict
        session metadata
    unique_id: str
        experiment id from data description
    debug: bool

    Returns
    -------
    h5_file: str
        path to h5 file
    output_dir: Path
        output directory
    reference_image_fp: str
        path to reference image
    """
    if not h5_file.is_file():
        h5_file = [f for f in h5_file.rglob("*.h5") if unique_id in str(f)][0]
    output_dir = make_output_directory(output_dir, unique_id)
    reference_image_fp = generate_single_plane_reference(h5_file, session)
    if debug:
        stem = h5_file.stem
        debug_file = Path("../scratch") / f"{stem}_debug.h5"
        with h5py.File(h5_file, "r") as f:
            data = f["data"][:5000]
            trial_locations = f["trial_locations"][()]
            epoch_filenames = f["epoch_locations"][()]
        with h5py.File(debug_file, "a") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("trial_locations", data=trial_locations)
            f.create_dataset("epoch_locations", data=epoch_filenames)
        h5_file = debug_file
    with h5py.File(h5_file, "r") as f:
        trial_locations = json.loads(f["trial_locations"][:][0])
        epoch_locations = json.loads(f["epoch_locations"][:][0])
    with open(output_dir / "trial_locations.json", "w") as j:
        json.dump(trial_locations, j)
    with open(output_dir / "epoch_locations.json", "w") as j:
        json.dump(epoch_locations, j)

    return str(h5_file), output_dir, str(reference_image_fp)


def get_frame_rate(session: dict):
    """Attempt to pull frame rate from session.json
    Returns none if frame rate not in session.json

    Parameters
    ----------
    session: dict
        session metadata

    Returns
    -------
    frame_rate: float
        frame rate in Hz
    """
    frame_rate_hz = None
    for i in session.get("data_streams", ""):
        if i.get("ophys_fovs", ""):
            frame_rate_hz = i["ophys_fovs"][0]["frame_rate"]
            break
    if isinstance(frame_rate_hz, str):
        frame_rate_hz = float(frame_rate_hz)
    return frame_rate_hz


class Suite2pMotionCorrection(BaseProcessor):
    """Suite2p-based motion correction processor.

    This class provides motion correction functionality using the Suite2p
    registration algorithm, with additional quality control and visualization
    features.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        settings: Optional[MotionCorrectionSettings] = None,
        debug: bool = False,
        **kwargs,
    ):
        """Initialize the Suite2p motion correction processor.

        Parameters
        ----------
        input_dir : Path
            Input directory containing data to process
        output_dir : Path
            Output directory for processed results
        settings : Optional[MotionCorrectionSettings]
            Motion correction settings. If None, will be parsed from command line
        debug : bool
            Whether to run in debug mode (process only subset of data)
        **kwargs
            Additional keyword arguments
        """
        super().__init__(input_dir, output_dir, **kwargs)

        # Parse settings if not provided
        if settings is None:
            self.settings = MotionCorrectionSettings()
        else:
            self.settings = settings

        self.debug = debug
        self.session = None
        self.data_description = None
        self.subject = None
        self.frame_rate_hz = None
        self.reference_image_fp = ""
        self.unique_id = ""
        self.input_file = ""

    def _load_metadata(self) -> None:
        """Load session, data description, and subject metadata."""
        session_fp = next(self.input_dir.rglob("session.json"))
        description_fp = next(self.input_dir.rglob("data_description.json"))
        subject_fp = next(self.input_dir.rglob("subject.json"))

        with open(session_fp, "r") as j:
            self.session = json.load(j)
        with open(description_fp, "r") as j:
            self.data_description = json.load(j)
        with open(subject_fp, "r") as j:
            self.subject = json.load(j)

        # Extract frame rate
        self.frame_rate_hz = get_frame_rate(self.session)

    def _setup_data_processing(self) -> None:
        """Set up data processing based on data type."""
        if self.settings.data_type == "TIFF":
            self.unique_id = "plane_0"
            try:
                self.input_file = next(self.input_dir.rglob("*/pophys"))
            except StopIteration:
                self.input_file = next(self.input_dir.rglob("pophys"))
            self.output_dir = make_output_directory(self.output_dir, self.unique_id)

        elif self.settings.data_type.lower() == "h5":
            self.unique_id = "MOp2_3_0"  # TODO: remove when upgrade to data-schema v2
            if "Bergamo" in self.session.get("rig_id", ""):
                h5_file, output_dir, reference_image_fp = singleplane_motion_correction(
                    self.input_dir, self.output_dir, self.session, self.unique_id, debug=self.debug
                )
                self.input_file = h5_file
                self.output_dir = output_dir
                self.reference_image_fp = reference_image_fp
            else:
                self.unique_id = "_".join(str(self.data_description["name"]).split("_")[-3:])
                h5_file, output_dir, frame_rate_hz = multiplane_motion_correction(
                    self.input_dir, self.output_dir, debug=self.debug
                )
                self.input_file = str(h5_file)
                self.output_dir = output_dir
                self.frame_rate_hz = frame_rate_hz
        else:
            raise ValueError(f"Data type {self.settings.data_type} not supported. " "Please use 'TIFF' or 'h5'.")

    def _setup_arguments(self) -> tuple[dict, dict]:
        """Set up processing arguments and suite2p configuration.

        Returns
        -------
        tuple[dict, dict]
            Tuple of (args, suite2p_args) dictionaries
        """
        # Convert settings to dictionary
        args = vars(self.settings)
        args["input_dir"] = str(args["input_dir"])
        args["output_dir"] = str(args["output_dir"])

        if not self.frame_rate_hz:
            self.frame_rate_hz = self.settings.frame_rate
            self.logger.warning("User input frame rate used. %s", self.frame_rate_hz)

        args["refImg"] = []
        if self.reference_image_fp:
            args["refImg"] = [self.reference_image_fp]

        # Construct output paths
        if self.settings.data_type == "TIFF":
            basename = self.unique_id
        else:
            basename = os.path.basename(self.input_file)

        args["movie_frame_rate_hz"] = self.frame_rate_hz

        # Set up output file paths
        for key, default in (
            ("motion_corrected_output", "_registered.h5"),
            ("motion_diagnostics_output", "_motion_transform.csv"),
            ("max_projection_output", "_maximum_projection.png"),
            ("avg_projection_output", "_average_projection.png"),
            ("registration_summary_output", "_registration_summary.png"),
            ("motion_correction_preview_output", "_motion_preview.webm"),
            ("output_json", "_motion_correction_output.json"),
        ):
            args[key] = os.path.join(str(self.output_dir), os.path.splitext(basename)[0] + default)

        # Hardcoded parameters
        args["movie_lower_quantile"] = 0.1
        args["movie_upper_quantile"] = 0.999
        args["preview_frame_bin_seconds"] = 2.0
        args["preview_playback_factor"] = 10.0
        args["n_batches"] = 20
        args["smooth_sigma_min"] = 0.65
        args["smooth_sigma_max"] = 2.15
        args["smooth_sigma_steps"] = 4
        args["smooth_sigma_time_min"] = 0
        args["smooth_sigma_time_max"] = 6
        args["smooth_sigma_time_steps"] = 7

        # Set up suite2p arguments
        suite2p_args = suite2p.default_ops()

        if self.settings.data_type == "h5":
            suite2p_args["h5py"] = str(self.input_file)
        else:
            suite2p_args["data_path"] = str(self.input_file)
            suite2p_args["look_one_level_down"] = True
            suite2p_args["tiff_list"] = [str(i) for i in self.input_file.glob("*.tif*")]

        # Configure suite2p parameters
        suite2p_args["roidetect"] = False
        suite2p_args["do_registration"] = self.settings.do_registration
        suite2p_args["align_by_chan"] = self.settings.align_by_chan
        suite2p_args["reg_tif"] = False
        suite2p_args["nimg_init"] = 500
        suite2p_args["maxregshift"] = self.settings.maxregshift
        suite2p_args["maxregshiftNR"] = self.settings.maxregshiftNR
        suite2p_args["batch_size"] = self.settings.batch_size

        if suite2p_args.get("h5py", ""):
            suite2p_args["h5py_key"] = "data"

        suite2p_args["smooth_sigma"] = self.settings.smooth_sigma
        suite2p_args["smooth_sigma_time"] = self.settings.smooth_sigma_time
        suite2p_args["nonrigid"] = self.settings.nonrigid
        suite2p_args["block_size"] = self.settings.block_size
        suite2p_args["snr_thresh"] = self.settings.snr_thresh
        suite2p_args["refImg"] = args["refImg"]
        suite2p_args["force_refImg"] = args["force_refImg"]

        return args, suite2p_args

    def _preprocess_data(self, args: dict, suite2p_args: dict) -> None:
        """Preprocess data including S3 copying and datatype checks."""
        # Copy from S3 if needed
        if suite2p_args.get("h5py", ""):
            if is_S3(suite2p_args["h5py"]):
                dst = "/scratch/" + Path(suite2p_args["h5py"]).name
                self.logger.info(f"copying {suite2p_args['h5py']} from S3 bucket to {dst}")
                shutil.copy(suite2p_args["h5py"], dst)
                suite2p_args["h5py"] = dst

        # Check data types
        if suite2p_args.get("tiff_list", ""):
            check_and_warn_on_datatype(
                filepath=suite2p_args["tiff_list"][0],
                logger=self.logger.warning,
                filetype="tiff",
            )
        else:
            check_and_warn_on_datatype(
                filepath=suite2p_args["h5py"],
                logger=self.logger.warning,
                filetype="h5",
                h5py_key=suite2p_args["h5py_key"],
            )

        # Handle empty frame removal
        if args["auto_remove_empty_frames"]:
            self.logger.info("Attempting to find empty frames at the start and end of the movie.")
            if suite2p_args.get("tiff_list", ""):
                lowside, highside = find_movie_start_end_empty_frames(
                    filepath=suite2p_args["tiff_list"],
                    logger=self.logger.warning,
                )
            else:
                lowside, highside = find_movie_start_end_empty_frames(
                    filepath=suite2p_args["h5py"],
                    h5py_key=suite2p_args["h5py_key"],
                    logger=self.logger.warning,
                )
            args["trim_frames_start"] = lowside
            args["trim_frames_end"] = highside
            self.logger.info(f"Found ({lowside}, {highside}) at the start/end of the movie.")

        # Handle reference image
        if suite2p_args["force_refImg"] and len(suite2p_args["refImg"]) == 0:
            suite2p_args, args = update_suite2p_args_reference_image(
                suite2p_args,
                args,
            )
        if self.reference_image_fp:
            suite2p_args, args = update_suite2p_args_reference_image(
                suite2p_args, args, reference_image_fp=self.reference_image_fp
            )

    def _run_suite2p_registration(self, suite2p_args: dict) -> tuple[str, str]:
        """Run suite2p registration and return paths to output files.

        Returns
        -------
        tuple[str, str]
            Tuple of (bin_path, ops_path)
        """
        self.logger.info(f"attempting to motion correct {suite2p_args['h5py']}")

        # Create temporary directory for Suite2P
        tmp_dir = tempfile.TemporaryDirectory(dir=self.settings.tmp_dir)
        tdir = tmp_dir.name
        suite2p_args["save_path0"] = tdir
        self.logger.info(f"Running Suite2P with output going to {tdir}")

        # Log suite2p arguments (excluding refImg which can't be serialized)
        copy_of_args = copy.deepcopy(suite2p_args)
        copy_of_args.pop("refImg")

        msg = f"running Suite2P v{suite2p.version} with args\n"
        msg += f"{json.dumps(copy_of_args, indent=2, sort_keys=True)}\n"
        self.logger.info(msg)

        if suite2p_args["force_refImg"]:
            self.logger.info(f"\tUsing custom reference image: {suite2p_args['refImg']}")

        if suite2p_args.get("h5py", ""):
            suite2p_args["h5py"] = suite2p_args["h5py"]

        # Run Suite2P
        suite2p.run_s2p(suite2p_args)

        # Get output paths
        bin_path = list(Path(tdir).rglob("data.bin"))[0]
        ops_path = list(Path(tdir).rglob("ops.npy"))[0]

        return str(bin_path), str(ops_path)

    def _process_registration_output(self, args: dict, suite2p_args: dict, bin_path: str, ops_path: str) -> np.ndarray:
        """Process suite2p registration output and create motion corrected movie.

        Returns
        -------
        np.ndarray
            Motion corrected movie data
        """
        # Load suite2p output
        ops = np.load(ops_path, allow_pickle=True).item()

        # Process outliers
        detrend_size = int(self.frame_rate_hz * args["outlier_detrend_window"])
        xlimit = int(ops["Lx"] * args["outlier_maxregshift"])
        ylimit = int(ops["Ly"] * args["outlier_maxregshift"])

        self.logger.info(
            "checking whether to clip where median-filtered "
            "offsets exceed (x,y) limits of "
            f"({xlimit},{ylimit}) [pixels]"
        )

        delta_x, x_clipped = identify_and_clip_outliers(np.array(ops["xoff"]), detrend_size, xlimit)
        delta_y, y_clipped = identify_and_clip_outliers(np.array(ops["yoff"]), detrend_size, ylimit)
        clipped_indices = list(set(x_clipped).union(set(y_clipped)))

        self.logger.info(f"{len(x_clipped)} frames clipped in x")
        self.logger.info(f"{len(y_clipped)} frames clipped in y")
        self.logger.info(f"{len(clipped_indices)} frames will be adjusted for clipping")

        # Load and process data
        data = suite2p.io.BinaryFile(ops["Ly"], ops["Lx"], bin_path).data

        if args["clip_negative"]:
            data[data < 0] = 0
            data = np.uint16(data)

        # Apply clipping corrections
        if not suite2p_args["nonrigid"]:
            for frame_index in clipped_indices:
                dx = delta_x[frame_index] - ops["xoff"][frame_index]
                dy = delta_y[frame_index] - ops["yoff"][frame_index]
                data[frame_index] = suite2p.registration.rigid.shift_frame(data[frame_index], dy, dx)

        # Reset empty frames
        reset_frame_shift(
            data,
            delta_y,
            delta_x,
            args["trim_frames_start"],
            args["trim_frames_end"],
        )

        # Create validity mask
        is_valid = np.ones(len(data), dtype="bool")
        is_valid[: args["trim_frames_start"]] = False
        is_valid[len(data) - args["trim_frames_end"] :] = False

        # Save motion corrected data
        self._save_motion_corrected_data(args, suite2p_args, data, ops)

        # Create and save motion diagnostics
        self._create_motion_diagnostics(args, suite2p_args, ops, delta_x, delta_y, is_valid, clipped_indices)

        return data

    def _save_motion_corrected_data(self, args: dict, suite2p_args: dict, data: np.ndarray, ops: dict) -> None:
        """Save motion corrected data to HDF5 file."""
        with h5py.File(args["motion_corrected_output"], "w") as f:
            f.create_dataset("data", data=data, chunks=(1, *data.shape[1:]))
            f.create_dataset("ref_image", data=suite2p_args["refImg"])

            # Save metadata
            args_copy = copy.deepcopy(args)
            suite_args_copy = copy.deepcopy(suite2p_args)
            suite_args_copy.pop("refImg")
            args_copy.pop("refImg")
            args_copy["suite2p_args"] = suite_args_copy
            f.create_dataset(name="metadata", data=json.dumps(args_copy).encode("utf-8"))

            # Save registration metrics
            f.create_group("reg_metrics")
            f.create_dataset("reg_metrics/regDX", data=ops.get("regDX", []))
            f.create_dataset("reg_metrics/regPC", data=ops.get("regPC", []))
            f.create_dataset("reg_metrics/tPC", data=ops.get("tPC", []))

        self.logger.info(f"saved Suite2P output to {args['motion_corrected_output']}")

    def _create_motion_diagnostics(
        self,
        args: dict,
        suite2p_args: dict,
        ops: dict,
        delta_x: np.ndarray,
        delta_y: np.ndarray,
        is_valid: np.ndarray,
        clipped_indices: list,
    ) -> None:
        """Create and save motion diagnostics CSV file."""
        if suite2p_args["nonrigid"]:
            # Convert nonrigid data to string for CSV storage
            nonrigid_x = [
                np.array2string(
                    arr,
                    separator=",",
                    suppress_small=True,
                    max_line_width=4096,
                )
                for arr in ops["xoff1"]
            ]
            nonrigid_y = [
                np.array2string(
                    arr,
                    separator=",",
                    suppress_small=True,
                    max_line_width=4096,
                )
                for arr in ops["yoff1"]
            ]
            nonrigid_corr = [
                np.array2string(
                    arr,
                    separator=",",
                    suppress_small=True,
                    max_line_width=4096,
                )
                for arr in ops["corrXY1"]
            ]
            motion_offset_df = pd.DataFrame(
                {
                    "framenumber": list(range(ops["nframes"])),
                    "x": ops["xoff"],
                    "y": ops["yoff"],
                    "x_pre_clip": ops["xoff"],
                    "y_pre_clip": ops["yoff"],
                    "correlation": ops["corrXY"],
                    "is_valid": is_valid,
                    "nonrigid_x": nonrigid_x,
                    "nonrigid_y": nonrigid_y,
                    "nonrigid_corr": nonrigid_corr,
                }
            )
        else:
            motion_offset_df = pd.DataFrame(
                {
                    "framenumber": list(range(ops["nframes"])),
                    "x": delta_x,
                    "y": delta_y,
                    "x_pre_clip": ops["xoff"],
                    "y_pre_clip": ops["yoff"],
                    "correlation": ops["corrXY"],
                    "is_valid": is_valid,
                }
            )

        motion_offset_df.to_csv(path_or_buf=args["motion_diagnostics_output"], index=False)
        self.logger.info(
            f"Writing the LIMS expected 'OphysMotionXyOffsetData' " f"csv file to: {args['motion_diagnostics_output']}"
        )

        if len(clipped_indices) != 0 and not suite2p_args["nonrigid"]:
            self.logger.warning(
                "some offsets have been clipped and the values "
                "for 'correlation' in "
                f"{args['motion_diagnostics_output']} "
                "where (x_clipped OR y_clipped) = True are not valid"
            )

    def _create_visualizations(self, args: dict, data: np.ndarray) -> None:
        """Create and save visualization outputs."""
        # Create projections
        mx_proj = projection_process(data, projection="max")
        av_proj = projection_process(data, projection="avg")

        # Save projections
        for im, dst_path in zip(
            [mx_proj, av_proj],
            [args["max_projection_output"], args["avg_projection_output"]],
        ):
            with Image.fromarray(im) as pilim:
                pilim.save(dst_path)
            self.logger.info(f"wrote {dst_path}")

        # Create summary PNG
        motion_offset_df = pd.read_csv(args["motion_diagnostics_output"])
        png_out_path = make_png(
            Path(args["max_projection_output"]),
            Path(args["avg_projection_output"]),
            motion_offset_df,
            Path(args["registration_summary_output"]),
        )
        self.logger.info(f"wrote {png_out_path}")

        # Create nonrigid summary if applicable
        if "nonrigid_x" in motion_offset_df.keys():
            p = Path(args["registration_summary_output"])
            nonrigid_png_out_path = make_nonrigid_png(
                Path(args["motion_corrected_output"]),
                Path(args["avg_projection_output"]),
                motion_offset_df,
                p.parent.joinpath(p.stem + "_nonrigid" + p.suffix),
            )
            self.logger.info(f"wrote {nonrigid_png_out_path}")

    def _create_preview_video(self, args: dict, suite2p_args: dict) -> None:
        """Create preview video of motion correction."""
        # Downsample and normalize movies
        ds_partial = partial(
            downsample_normalize,
            frame_rate=args["movie_frame_rate_hz"],
            bin_size=args["preview_frame_bin_seconds"],
            lower_quantile=args["movie_lower_quantile"],
            upper_quantile=args["movie_upper_quantile"],
        )

        if suite2p_args.get("h5py", ""):
            h5_file = suite2p_args["h5py"]
            processed_vids = [
                ds_partial(i)
                for i in [
                    Path(h5_file),
                    Path(args["motion_corrected_output"]),
                ]
            ]
        else:
            tiff_array = tiff_to_numpy(suite2p_args["tiff_list"])
            processed_vids = [
                ds_partial(i)
                for i in [
                    tiff_array,
                    Path(args["motion_corrected_output"]),
                ]
            ]

        self.logger.info("finished downsampling motion corrected and non-motion corrected movies")

        # Create tiled video
        try:
            tiled_vids = np.block(processed_vids)
            playback_fps = args["preview_playback_factor"] / args["preview_frame_bin_seconds"]
            encode_video(tiled_vids, args["motion_correction_preview_output"], playback_fps)
            self.logger.info("wrote " f"{args['motion_correction_preview_output']}")
        except Exception:
            self.logger.info("Could not write motion correction preview")

    def _compute_additional_metrics(self, args: dict, suite2p_args: dict) -> None:
        """Compute additional registration metrics."""
        if suite2p_args.get("h5py", ""):
            with (
                h5py.File(suite2p_args["h5py"]) as f_raw,
                h5py.File(args["motion_corrected_output"], "r+") as f,
            ):
                mov_raw = f_raw["data"]
                mov = f["data"]
                regDX = f["reg_metrics/regDX"][:]
                crispness = compute_crispness(mov_raw, mov)
                self.logger.info("computed crispness of mean image before and after registration")

                # Compute residual optical flow
                if f["reg_metrics/regPC"][:].any():
                    flows, farnebackDX = compute_residual_optical_flow(f["reg_metrics/regPC"])
                    f.create_dataset("reg_metrics/farnebackROF", data=flows)
                    f.create_dataset("reg_metrics/farnebackDX", data=farnebackDX)
                    self.logger.info("computed residual optical flow of top PCs using Farneback method")

                f.create_dataset("reg_metrics/crispness", data=crispness)
                self.logger.info("appended additional registration metrics to" f"{args['motion_corrected_output']}")
        else:
            tiff_array = tiff_to_numpy(suite2p_args["tiff_list"])
            with h5py.File(args["motion_corrected_output"], "r+") as f:
                regDX = f["reg_metrics/regDX"][:]
                crispness = compute_crispness(tiff_array, f["data"])
                self.logger.info("computed crispness of mean image before and after registration")

                if f["reg_metrics/regPC"][:].any():
                    regPC = f["reg_metrics/regPC"]
                    flows, farnebackDX = compute_residual_optical_flow(regPC)
                    f.create_dataset("reg_metrics/farnebackROF", data=flows)
                    f.create_dataset("reg_metrics/farnebackDX", data=farnebackDX)
                    self.logger.info("computed residual optical flow of top PCs using Farneback method")

                f.create_dataset("reg_metrics/crispness", data=crispness)
                self.logger.info("appended additional registration metrics to" f"{args['motion_corrected_output']}")

        # Create flow visualization
        with h5py.File(args["motion_corrected_output"], "r") as f:
            if f["reg_metrics/regDX"][:].any() and f["reg_metrics/farnebackDX"][:].any():
                regDX = f["reg_metrics/regDX"][:]
                farnebackDX = f["reg_metrics/farnebackDX"][:]
                for iPC in set(
                    (
                        np.argmax(regDX[:, -1]),
                        np.argmax(farnebackDX[:, -1]),
                    )
                ):
                    p = Path(args["registration_summary_output"])
                    flow_png(
                        Path(args["motion_corrected_output"]),
                        str(p.parent / p.stem),
                        iPC,
                    )
                self.logger.info(f"created images of PC_low, PC_high, and PC_rof for PC {iPC}")

    def _write_metadata(self, args: dict) -> None:
        """Write processing metadata and QC metrics."""
        if self.settings.data_type == "TIFF":
            input_file = self.input_file[0]
        else:
            input_file = self.input_file

        basename = os.path.basename(input_file)

        # Write data process metadata
        write_data_process(
            args,
            input_file,
            args["motion_corrected_output"],
            self.output_dir,
            basename.split(".")[0],
            self.start_time,
            end_time=dt.now(),
        )

        # Serialize QC metrics
        serialize_registration_summary_qcmetric(self.output_dir)
        serialize_fov_quality_qcmetric(self.output_dir)

    def run(self) -> Dict[str, Any]:
        """Execute the Suite2p motion correction pipeline.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing processing results and metadata
        """
        try:
            self.logger.info("Starting Suite2p motion correction")

            # Load metadata
            self._load_metadata()

            # Setup processing
            self._setup_data_processing()

            # Setup arguments
            args, suite2p_args = self._setup_arguments()

            # Preprocess data
            self._preprocess_data(args, suite2p_args)

            # Run suite2p registration
            bin_path, ops_path = self._run_suite2p_registration(suite2p_args)

            # Process registration output
            data = self._process_registration_output(args, suite2p_args, bin_path, ops_path)

            # Create visualizations
            self._create_visualizations(args, data)

            # Create preview video
            self._create_preview_video(args, suite2p_args)

            # Compute additional metrics
            self._compute_additional_metrics(args, suite2p_args)

            # Write metadata
            self._write_metadata(args)

            self._finalize()

            return {
                "status": "success",
                "motion_corrected_output": args["motion_corrected_output"],
                "motion_diagnostics_output": args["motion_diagnostics_output"],
                "max_projection_output": args["max_projection_output"],
                "avg_projection_output": args["avg_projection_output"],
                "registration_summary_output": args["registration_summary_output"],
                "motion_correction_preview_output": args["motion_correction_preview_output"],
                "metadata": self.get_processing_metadata(),
            }

        except Exception as e:
            self.logger.error(f"Motion correction failed: {str(e)}")
            self._finalize()
            return {"status": "failed", "error": str(e), "metadata": self.get_processing_metadata()}


if __name__ == "__main__":  # pragma: nocover
    # Command-line interface for backwards compatibility
    try:
        from aind_log_utils.log import setup_logging

        has_aind_log = True
    except ImportError:
        has_aind_log = False

    # Set the log level and name the logger
    logger = logging.getLogger("Suite2P motion correction")
    logger.setLevel(logging.INFO)

    # Parse command-line arguments
    parser = MotionCorrectionSettings()

    # Setup logging
    session_fp = next(parser.input_dir.rglob("session.json"))
    description_fp = next(parser.input_dir.rglob("data_description.json"))
    subject_fp = next(parser.input_dir.rglob("subject.json"))

    with open(description_fp, "r") as j:
        data_description = json.load(j)
    with open(subject_fp, "r") as j:
        subject = json.load(j)

    subject_id = subject.get("subject_id", "")
    name = data_description.get("name", "")
    if has_aind_log:
        setup_logging("aind-ophys-motion-correction", mouse_id=subject_id, session_name=name)

    # Create and run processor
    processor = Suite2pMotionCorrection(
        input_dir=parser.input_dir, output_dir=parser.output_dir, settings=parser, debug=parser.debug, logger=logger
    )

    result = processor.run()

    if result["status"] == "success":
        logger.info("Motion correction completed successfully")
    else:
        logger.error(f"Motion correction failed: {result.get('error', 'Unknown error')}")
        exit(1)
