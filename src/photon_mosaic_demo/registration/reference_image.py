"""Reference image creation and optimization module for suite2p motion correction.

This module contains functions for creating reference images used in motion correction,
optimizing motion correction parameters, and related utilities.
"""

import json
import logging
import warnings
from functools import lru_cache
from itertools import product
from pathlib import Path
from time import time
from typing import Callable, List, Optional, Union

import h5py
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
from scipy.stats import sigmaclip
from suite2p.registration.register import pick_initial_reference
from suite2p.registration.rigid import (
    apply_masks,
    compute_masks,
    phasecorr,
    phasecorr_reference,
    shift_frame,
)


def h5py_to_numpy(
    h5py_file: str,
    h5py_key: str,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
) -> np.ndarray:
    """Converts a h5py dataset to a numpy array

    Parameters
    ----------
    h5py_file: str
        h5py file path
    h5py_key : str
        key to the dataset
    trim_frames_start : int
        Number of frames to disregard from the start of the movie. Default 0.
    trim_frames_end : int
        Number of frames to disregard from the end of the movie. Default 0.
    Returns
    -------
    np.ndarray
        numpy array
    """
    with h5py.File(h5py_file, "r") as f:
        n_frames = f[h5py_key].shape[0]
        if trim_frames_start > 0 or trim_frames_end > 0:
            return f[h5py_key][trim_frames_start : n_frames - trim_frames_end]
        else:
            return f[h5py_key][:]


@lru_cache(maxsize=None)
def _tiff_to_numpy(tiff_file: Path) -> np.ndarray:
    with ScanImageTiffReader(tiff_file) as reader:
        return reader.data()


def tiff_to_numpy(
    tiff_list: List[Path], trim_frames_start: int = 0, trim_frames_end: int = 0
) -> np.ndarray:
    """
    Converts a list of TIFF files to a single numpy array, with optional frame trimming.

    Parameters
    ----------
    tiff_list : List[str]
        List of TIFF file paths to process
    trim_frames_start : int, optional
        Number of frames to remove from the start (default: 0)
    trim_frames_end : int, optional
        Number of frames to remove from the end (default: 0)

    Returns
    -------
    np.ndarray
        Combined array of all TIFF data with specified trimming

    Raises
    ------
    ValueError
        If trim values exceed total number of frames or are negative
    """
    if trim_frames_start < 0 or trim_frames_end < 0:
        raise ValueError("Trim values must be non-negative")

    def get_total_frames(tiff_files: List[Path]) -> int:
        """Get total number of frames across all TIFF files."""
        total = 0
        for tiff_path in tiff_files:
            array = _tiff_to_numpy(tiff_path)
            total += array.shape[0]
        return total

    # Validate trim parameters
    total_frames = get_total_frames(tiff_list)
    if trim_frames_start + trim_frames_end >= total_frames:
        raise ValueError(
            f"Trim values ({trim_frames_start} + {trim_frames_end} = "
            f"{trim_frames_start + trim_frames_end}) exceed total frames ({total_frames})"
        )

    # Initialize variables for frame counting
    processed_frames = 0
    arrays_to_stack = []

    for tiff_path in tiff_list:
        array = _tiff_to_numpy(tiff_path)
        file_frames = array.shape[0]
        
        # Determine which frames to include from this file
        start_frame = max(0, trim_frames_start - processed_frames)
        end_frame = min(
            file_frames, 
            file_frames - max(0, (processed_frames + file_frames) - (total_frames - trim_frames_end))
        )
        
        if start_frame < end_frame:
            arrays_to_stack.append(array[start_frame:end_frame])
        
        processed_frames += file_frames

    # Stack all arrays along the appropriate axis
    if not arrays_to_stack:
        raise ValueError("No frames remaining after trimming")

    return (
        np.concatenate(arrays_to_stack, axis=0)
        if len(arrays_to_stack) > 1
        else arrays_to_stack[0]
    )


def load_initial_frames(
    file_path: Union[str, list],
    h5py_key: str,
    n_frames: int,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
) -> np.ndarray:
    """Load a subset of frames from the hdf5 data specified by file_path.

    Only loads frames between trim_frames_start and n_frames - trim_frames_end
    from the movie. If both are 0, load frames from the full movie.

    Parameters
    ----------
    file_path : str
        Location of the raw ophys, HDF5 data to load.
    h5py_key : str
        Name of the dataset to load from the HDF5 file.
    n_frames : int
        Number of frames to load from the input HDF5 data.

    Returns
    -------
    frames : array-like, (n_frames, nrows, ncols)
        Frames selected from the input raw data linearly spaced in index of the
        time axis. If n_frames > tot_frames, a number of frames equal to
        tot_frames is returned.
    """
    if isinstance(file_path, str):
        array = h5py_to_numpy(file_path, h5py_key, trim_frames_start, trim_frames_end)
    elif isinstance(file_path, list):
        array = tiff_to_numpy(file_path, trim_frames_start, trim_frames_end)
    else:
        raise ValueError("File type not supported")
    # Total number of frames in the movie.
    tot_frames = array.shape[0]
    requested_frames = np.linspace(
        0, tot_frames, 1 + min(n_frames, tot_frames), dtype=int
    )[:-1]
    frames = array[requested_frames]
    return frames


def remove_extrema_frames(input_frames: np.ndarray, n_sigma: float = 3) -> np.ndarray:
    """Remove frames with extremum mean values from the frames used in
    reference image processing/creation.

    Likely these are empty frames of pure noise or very high intensity frames
    relative to mean.

    Parameters
    ----------
    input_frames : numpy.ndarray, (N, M, K)
        Set of frames to trim.
    n_sigma : float, optional
        Number of standard deviations to above which to clip. Default is 3
        which was found to remove all empty frames while preserving most
        frames.

    Returns
    -------
    trimmed_frames : numpy.ndarray, (N, M, K)
        Set of frames with the extremum frames removed.
    """
    frame_means = np.mean(input_frames, axis=(1, 2))
    _, low_cut, high_cut = sigmaclip(frame_means, low=n_sigma, high=n_sigma)
    trimmed_frames = input_frames[
        np.logical_and(frame_means > low_cut, frame_means < high_cut)
    ]
    return trimmed_frames


def compute_reference(
    input_frames: np.ndarray,
    niter: int,
    maxregshift: float,
    smooth_sigma: float,
    smooth_sigma_time: float,
    mask_slope_factor: float = 3,
) -> np.ndarray:
    """Computes a stacked reference image from the input frames.

    Modified version of Suite2P's compute_reference function with no updating
    of input frames. Picks initial reference then iteratively aligns frames to
    create reference. This code does not reproduce the pre-processing suite2p
    does to data from 1Photon scopes. As such, if processing 1Photon data, the
    user should use the suite2p reference image creation.

    Parameters
    ----------
    input_frames : array-like, (n_frames, nrows, ncols)
        Set of frames to create a reference from.
    niter : int
        Number of iterations to perform when creating the reference image.
    maxregshift : float
        Maximum shift allowed as a fraction of the image width or height, which
        ever is longer.
    smooth_sigma : float
        Width of the Gaussian used to smooth the phase correlation between the
        reference and the frame with which it is being registered.
    smooth_sigma_time : float
        Width of the Gaussian used to smooth between multiple frames by before
        phase correlation.
    mask_slope_factor : int
        Factor to multiply ``smooth_sigma`` by when creating masks for the
        reference image during suite2p phase correlation. These masks down
        weight edges of the image. The default used in suite2p, where this
        method is adapted from, is 3.

    Returns
    -------
    refImg : array-like, (nrows, ncols)
        Reference image created from the input data.
    """
    # Get the dtype of the input frames to properly cast the final reference
    # image as the same type.
    frames_dtype = input_frames.dtype
    # Get initial reference image from suite2p.
    frames = remove_extrema_frames(input_frames)
    ref_image = pick_initial_reference(frames)

    # Determine how much to pad our frames by before shifting to prevent
    # wraps.
    pad_y = int(np.ceil(maxregshift * ref_image.shape[0]))
    pad_x = int(np.ceil(maxregshift * ref_image.shape[1]))

    for idx in range(niter):
        # Compute the number of frames to select in creating the reference
        # image. At most we select half to the input frames.
        nmax = int(frames.shape[0] * (1.0 + idx) / (2 * niter))

        # rigid Suite2P phase registration.
        ymax, xmax, cmax = phasecorr(
            data=apply_masks(
                frames,
                *compute_masks(
                    refImg=ref_image,
                    maskSlope=mask_slope_factor * smooth_sigma,
                ),
            ),
            cfRefImg=phasecorr_reference(
                refImg=ref_image,
                smooth_sigma=smooth_sigma,
            ),
            maxregshift=maxregshift,
            smooth_sigma_time=smooth_sigma_time,
        )

        # Find the indexes of the frames that are the most correlated and
        # select the first nmax.
        isort = np.argsort(-cmax)[:nmax]

        # Copy the most correlated frames so we don't shift the original data.
        # We pad this data to prevent wraps from showing up in the reference
        # image. We pad with NaN values to enable us to use nanmean and only
        # average those pixels that contain data in the average.
        max_corr_frames = np.pad(
            array=frames[isort].astype(float),
            pad_width=((0, 0), (pad_y, pad_y), (pad_x, pad_x)),
            constant_values=np.nan,
        )
        max_corr_xmax = xmax[isort]
        max_corr_ymax = ymax[isort]
        # Apply shift to the copy of the frames.
        for frame, dy, dx in zip(max_corr_frames, max_corr_ymax, max_corr_xmax):
            frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)

        # Create a new reference image from the highest correlated data.
        with warnings.catch_warnings():
            # Assuming the motion correction went well, there should be a lot
            # of empty values in the padded area around the frames. We suppress
            # warnings for these "Empty Slices" as they are expected.
            warnings.filterwarnings("ignore", "Mean of empty slice")
            ref_image = np.nanmean(max_corr_frames, axis=0)
        # Shift reference image to position of mean shifts to remove any bulk
        # displacement.
        ref_image = shift_frame(
            frame=ref_image,
            dy=int(np.round(-max_corr_ymax.mean())),
            dx=int(np.round(-max_corr_xmax.mean())),
        )
        # Clip the reference image back down to the original size and remove
        # any NaNs remaining. Throw warning if a NaN is found.
        ref_image = ref_image[pad_y:-pad_y, pad_x:-pad_x]
        if np.any(np.isnan(ref_image)):
            # NaNs can sometimes be left over from the image padding during the
            # first few iterations before the reference image has converged.
            # If there are still NaNs left after the final iteration, we
            # throw the following warning.
            if idx + 1 == niter:
                logging.warning(
                    f"Warning: {np.isnan(ref_image).sum()} NaN pixels were "
                    "found in the reference image on the final iteration. "
                    "Likely the image quality is low and shifting frames "
                    "failed. Setting NaN values to the image mean."
                )
            ref_image = np.nan_to_num(ref_image, nan=np.nanmean(ref_image), copy=False)
        ref_image = ref_image.astype(frames_dtype)

    return ref_image


def compute_acutance(
    image: np.ndarray,
    min_cut_y: int = 0,
    max_cut_y: int = 0,
    min_cut_x: int = 0,
    max_cut_x: int = 0,
) -> float:
    """Compute the acutance (sharpness) of an image.

    Parameters
    ----------
    image : numpy.ndarray, (N, M)
        Image to compute acutance of.
    min_cut_y : int
        Number of pixels to cut from the beginning of the y axis.
    max_cut_y : int
        Number of pixels to cut from the end of the y axis.
    min_cut_x : int
        Number of pixels to cut from the beginning of the x axis.
    max_cut_x : int
        Number of pixels to cut from the end of the x axis.

    Returns
    -------
    acutance : float
        Acutance of the image.
    """
    im_max_y, im_max_x = image.shape

    cut_image = image[
        min_cut_y : im_max_y - max_cut_y, min_cut_x : im_max_x - max_cut_x
    ]
    grady, gradx = np.gradient(cut_image)
    return (grady**2 + gradx**2).mean()


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
        suite2p_args["bidiphase"] = 0
    if suite2p_args.get("nonrigid") is None:
        suite2p_args["nonrigid"] = False
    if suite2p_args.get("norm_frames") is None:
        suite2p_args["norm_frames"] = True
    # Don't use nonrigid for parameter search.
    suite2p_args["nonrigid"] = False


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
    from suite2p.registration.register import register_frames
    
    ave_frame = np.zeros((ref_image.shape[0], ref_image.shape[1]))
    min_y = 0
    max_y = 0
    min_x = 0
    max_x = 0
    tot_frames = input_frames.shape[0]
    add_modify_required_parameters(suite2p_args)
    for start_idx in np.arange(0, tot_frames, batch_size):
        end_idx = min(start_idx + batch_size, tot_frames)
        batch_data = input_frames[start_idx:end_idx]
        motion_output = register_frames(
            refImg=ref_image, frames=batch_data, **suite2p_args
        )
        ave_frame += motion_output["meanImg"] * batch_data.shape[0]
        # Record the maximum shift for each direction.
        min_y = min(min_y, motion_output["yoff"].min())
        max_y = max(max_y, motion_output["yoff"].max())
        min_x = min(min_x, motion_output["xoff"].min())
        max_x = max(max_x, motion_output["xoff"].max())
    ave_frame /= tot_frames

    return {
        "ave_image": ave_frame,
        "min_y": int(np.fabs(min_y)),
        "max_y": int(max_y),
        "min_x": int(np.fabs(min_x)),
        "max_x": int(max_x),
    }


def optimize_motion_parameters(
    initial_frames: np.ndarray,
    smooth_sigmas: np.array,
    smooth_sigma_times: np.array,
    suite2p_args: dict,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
    n_batches: int = 20,
    logger: Optional[Callable] = None,
) -> dict:
    """Loop over a range of parameters and select the best set from the
    max acutance of the final, average image.

    Parameters
    ----------
    initial_frames : numpy.ndarray, (N, M, K)
        Smaller subset of frames to create a reference image from.
    smooth_sigmas : numpy.ndarray, (N,)
        Array of suite2p smooth sigma values to attempt. Number of iterations
        will be len(`smooth_sigmas`) * len(`smooth_sigma_times`).
    smooth_sigma_times : numpy.ndarray, (N,)
        Array of suite2p smooth sigma time values to attempt. Number of
        iterations will be len(`smooth_sigmas`) * len(`smooth_sigma_times`).
    suite2p_args : dict
        A dictionary of suite2p configs containing at minimum:

        ``"h5py"``
            HDF5 file containing to the movie to motion correct.
        ``"h5py_key"``
            Name of the dataset where the movie to be motion corrected is
            stored.
        ``"maxregshift"``
            Maximum shift allowed as a fraction of the image dimensions.
    trim_frames_start : int, optional
        Number of frames to disregard from the start of the movie. Default 0.
    trim_frames_start : int, optional
        Number of frames to disregard from the end of the movie. Default 0.
    n_batches : int
        Number of batches to load. Processing a large number of frames at once
        will likely result in running out of memory, hence processing in
        batches. Total returned size isn_batches * suit2p_args['batch_size'].
    logger : Optional[Callable]
        Function to print to stdout or a log.

    Returns
    -------
    best_result : dict
        A dict containing the final results of the search:

        ``ave_image``
            Image created with the settings yielding the highest image acutance
            (numpy.ndarray, (N, M))
        ``ref_image``
            Reference Image created with the settings yielding the highest
            image acutance (numpy.ndarray, (N, M))
        ``acutance``
            Acutance of ``best_image``. (float)
        ``smooth_sigma``
            Value of ``smooth_sigma`` found to yield the best acutance (float).
        ``smooth_sigma_time``
            Value of ``smooth_sigma_time`` found to yield the best acutance
            (float).
    """
    best_results = {
        "acutance": 1e-16,
        "ave_image": np.array([]),
        "ref_image": np.array([]),
        "smooth_sigma": -1,
        "smooth_sigma_time": -1,
    }
    logger("Starting search for best smoothing parameters...")
    sub_frames = load_representative_sub_frames(
        suite2p_args["h5py"],
        suite2p_args["h5py_key"],
        trim_frames_start,
        trim_frames_end,
        n_batches=n_batches,
        batch_size=suite2p_args["batch_size"],
    )
    start_time = time()
    for param_spatial, param_time in product(smooth_sigmas, smooth_sigma_times):
        current_args = suite2p_args.copy()
        current_args["smooth_sigma"] = param_spatial
        current_args["smooth_sigma_time"] = param_time

        if logger:
            logger(
                f'\tTrying: smooth_sigma={current_args["smooth_sigma"]}, '
                f'smooth_sigma_time={current_args["smooth_sigma_time"]}'
            )

        ref_image = compute_reference(
            initial_frames,
            8,
            current_args["maxregshift"],
            current_args["smooth_sigma"],
            current_args["smooth_sigma_time"],
        )
        image_results = create_ave_image(
            ref_image,
            sub_frames.copy(),
            current_args,
            batch_size=suite2p_args["batch_size"],
        )
        ave_image = image_results["ave_image"]
        # Compute the acutance ignoring the motion border. Sharp motion
        # borders can potentially get rewarded with high acutance.
        current_acu = compute_acutance(
            ave_image,
            image_results["min_y"],
            image_results["max_y"],
            image_results["min_x"],
            image_results["max_x"],
        )

        if current_acu > best_results["acutance"]:
            best_results["acutance"] = current_acu
            best_results["ave_image"] = ave_image
            best_results["ref_image"] = ref_image
            best_results["smooth_sigma"] = current_args["smooth_sigma"]
            best_results["smooth_sigma_time"] = current_args["smooth_sigma_time"]
        if logger:
            logger(f"\t\tacutance: {current_acu}")
    end_time = time()
    if logger:
        logger(
            f"Parameter search completed in {end_time - start_time:.2f}s. "
            f'Best settings: smooth_sigma={best_results["smooth_sigma"]}, '
            f'smooth_sigma_time={best_results["smooth_sigma_time"]}, '
            f'acutance={best_results["acutance"]}'
        )
    return best_results


def generate_single_plane_reference(fp: Path, session) -> Path:
    """Generate virtual movies for Bergamo data

    Parameters
    ----------
    fp: Path
        path to h5 file
    session: dict
        session metadata
    Returns
    -------
    Path
        path to reference image
    """
    with h5py.File(fp, "r") as f:
        # take the first bci epoch to save out reference image TODO
        tiff_stems = json.loads(f["epoch_locations"][:][0])
        bci_epochs = [
            i
            for i in session["stimulus_epochs"]
            if i["stimulus_name"] == "single neuron BCI conditioning"
        ]
        bci_epoch_loc = [i["output_parameters"]["tiff_stem"] for i in bci_epochs][0]
        frame_length = tiff_stems[bci_epoch_loc][1] - tiff_stems[bci_epoch_loc][0]
        vsource = h5py.VirtualSource(f["data"])
        layout = h5py.VirtualLayout(
            shape=(frame_length, *f["data"].shape[1:]), dtype=f["data"].dtype
        )
        layout[0:frame_length] = vsource[
            tiff_stems[bci_epoch_loc][0] : tiff_stems[bci_epoch_loc][1]
        ]

        with h5py.File("../scratch/reference_image.h5", "w") as ref:

            ref.create_virtual_dataset("data", layout)
    return Path("../scratch/reference_image.h5")


def update_suite2p_args_reference_image(
    suite2p_args: dict, args: dict, reference_image_fp=None, logger=None
):
    """Update the suite2p_args dictionary with the reference image.

    Parameters
    ----------
    suite2p_args : dict
        Suite2p ops dictionary.
    args : dict
        Dictionary of arguments from the command line.
    reference_image_fp : Path
        Path to the reference image to use. Default is None.
    logger : Optional[Callable]
        Logger function for output. Default is None.

    Returns
    -------
    suite2p_args : dict
        Updated suite2p_args dictionary.
    args : dict
        Updated args dictionary.
    """
    # Use our own version of compute_reference to create the initial
    # reference image used by suite2p.
    if logger:
        logger.info(
            f'Loading {suite2p_args["nimg_init"]} frames ' "for reference image creation."
        )
    if reference_image_fp:
        initial_frames = load_initial_frames(
            file_path=reference_image_fp,
            h5py_key=suite2p_args["h5py_key"],
            n_frames=suite2p_args["nimg_init"],
            trim_frames_start=args["trim_frames_start"],
            trim_frames_end=args["trim_frames_end"],
        )

    else:
        if suite2p_args.get("h5py", None):
            file_path = suite2p_args["h5py"]
            h5py_key = suite2p_args["h5py_key"]
        else:
            file_path = suite2p_args["tiff_list"]
            h5py_key = None
        initial_frames = load_initial_frames(
            file_path=file_path,
            h5py_key=h5py_key,
            n_frames=suite2p_args["nimg_init"],
            trim_frames_start=args["trim_frames_start"],
            trim_frames_end=args["trim_frames_end"],
        )

    if args["do_optimize_motion_params"]:
        if logger:
            logger.info("Attempting to optimize registration parameters Using:")
            logger.info(
                "\tsmooth_sigma range: "
                f'{args["smooth_sigma_min"]} - '
                f'{args["smooth_sigma_max"]}, '
                f'steps: {args["smooth_sigma_steps"]}'
            )
            logger.info(
                "\tsmooth_sigma_time range: "
                f'{args["smooth_sigma_time_min"]} - '
                f'{args["smooth_sigma_time_max"]}, '
                f'steps: {args["smooth_sigma_time_steps"]}'
            )

        # Create linear spaced arrays for the range of smooth
        # parameters to try.
        smooth_sigmas = np.linspace(
            args["smooth_sigma_min"],
            args["smooth_sigma_max"],
            args["smooth_sigma_steps"],
        )
        smooth_sigma_times = np.linspace(
            args["smooth_sigma_time_min"],
            args["smooth_sigma_time_max"],
            args["smooth_sigma_time_steps"],
        )

        optimize_result = optimize_motion_parameters(
            initial_frames=initial_frames,
            smooth_sigmas=smooth_sigmas,
            smooth_sigma_times=smooth_sigma_times,
            suite2p_args=suite2p_args,
            trim_frames_start=args["trim_frames_start"],
            trim_frames_end=args["trim_frames_end"],
            n_batches=args["n_batches"],
            logger=logger.info if logger else None,
        )
        if args["use_ave_image_as_reference"]:
            suite2p_args["refImg"] = optimize_result["ave_image"]
        else:
            suite2p_args["refImg"] = optimize_result["ref_image"]
        suite2p_args["smooth_sigma"] = optimize_result["smooth_sigma"]
        suite2p_args["smooth_sigma_time"] = optimize_result["smooth_sigma_time"]
    else:
        # Create the initial reference image and store it in the
        # suite2p_args dictionary. 8 iterations is the current default
        # in suite2p.
        tic = -time()
        if logger:
            logger.info("Creating custom reference image...")
        suite2p_args["refImg"] = compute_reference(
            input_frames=initial_frames,
            niter=args["max_reference_iterations"],
            maxregshift=suite2p_args["maxregshift"],
            smooth_sigma=suite2p_args["smooth_sigma"],
            smooth_sigma_time=suite2p_args["smooth_sigma_time"],
        )
        tic += time()
        if logger:
            logger.info(f"took {tic}s")
    return suite2p_args, args
