"""Map ROI extractor implementations to BaseImaging and BaseRois"""

from pathlib import Path

from roiextractors.extractorlist import imaging_extractor_dict
from roiextractors.imagingextractor import ImagingExtractor

from ..core import BaseImaging, BaseImagingSegment


class BaseROIExtractorImaging(BaseImaging):
    """Adapter class that wraps roiextractors imaging classes to work with BaseImaging.

    This class allows you to load imaging data from various file formats (TIFF, ScanImage, etc.)
    using the roiextractors library, and use them with the BaseImaging interface.

    Parameters
    ----------
    imaging_name : str
        The name of the imaging extractor from roiextractors.imaging_extractor_dict.
        Examples: "ScanImageImagingExtractor", "TiffImagingExtractor", "BrukerTiffImagingExtractor", etc.
    **kwargs
        Additional keyword arguments passed to the roiextractors imaging class.
        These vary by format (e.g., file_path, channel_name, etc.)

    Examples
    --------
    Load a ScanImage TIFF file:
    >>> imaging = BaseROIExtractorImaging(
    ...     imaging_name="ScanImageImagingExtractor",
    ...     file_path="path/to/data.tif",
    ...     channel_name="Channel 1"
    ... )
    >>> frames = imaging.get_series(start_frame=0, end_frame=10)

    Load a standard TIFF file:
    >>> imaging = BaseROIExtractorImaging(
    ...     imaging_name="TiffImagingExtractor",
    ...     file_path="path/to/data.tif"
    ... )
    """

    def __init__(self, imaging_name: str, **kwargs):
        self.roiextractor_imaging_class = imaging_extractor_dict[imaging_name]

        roi_extractor = self.roiextractor_imaging_class(**kwargs)
        width, height = roi_extractor.get_sample_shape()

        segment = BaseROIExtractorImagingSegment(roi_extractor)
        BaseImaging.__init__(
            self,
            shape=(width, height),
            sampling_frequency=roi_extractor.get_sampling_frequency(),
        )
        self.add_imaging_segment(segment)
        self.name = f"{imaging_name} (ROIExtractors)"

        self._kwargs = {"imaging_name": imaging_name, **kwargs}


class BaseROIExtractorImagingSegment(BaseImagingSegment):
    """Base class for ROI extractors that work with BaseImaging data."""

    def __init__(self, roi_extractor_imaging: ImagingExtractor):
        BaseImagingSegment.__init__(
            self, sampling_frequency=roi_extractor_imaging.get_sampling_frequency()
        )
        self.roiextractor_extractor = roi_extractor_imaging

    def get_num_samples(self):
        return self.roiextractor_extractor.get_num_samples()

    def get_series(self, start_frame=None, end_frame=None):
        return self.roiextractor_extractor.get_series(start_frame, end_frame)


def auto_add_roiextractor_methods():
    """Automatically add all methods from ImagingExtractor to BaseROIExtractorImagingSegment."""
    for method_name in dir(ImagingExtractor):
        if not method_name.startswith("_") and callable(
            getattr(ImagingExtractor, method_name)
        ):
            if hasattr(BaseROIExtractorImagingSegment, method_name):
                continue

            def make_wrapper(method_name):
                def wrapper(self, *args, **kwargs):
                    return getattr(self.roiextractor_extractor, method_name)(
                        *args, **kwargs
                    )

                wrapper.__name__ = method_name
                return wrapper

            setattr(
                BaseROIExtractorImagingSegment, method_name, make_wrapper(method_name)
            )


def get_imaging_extractor(
    file_path: str, imaging_name: str | None = None, **kwargs
) -> BaseROIExtractorImaging:
    """Automatically detect and load imaging data from a file.

    This function attempts to identify the correct imaging extractor based on the file
    format and metadata. If imaging_name is provided, it will use that specific extractor.

    Parameters
    ----------
    file_path : str
        Path to the imaging data file.
    imaging_name : str, optional
        The name of the imaging extractor to use. If None, attempts automatic detection.
        Examples: "ScanImageImaging", "TiffImaging", "BrukerTiffImaging", etc.
    **kwargs
        Additional keyword arguments passed to the imaging extractor.

    Returns
    -------
    BaseROIExtractorImaging
        The loaded imaging data as a BaseImaging object.

    Examples
    --------
    Automatic detection:
    >>> imaging = read_imaging("path/to/data.tif")

    Specify extractor explicitly:
    >>> imaging = read_imaging("path/to/data.tif", imaging_name="ScanImageImaging")

    With additional parameters:
    >>> imaging = read_imaging("path/to/data.tif", channel_name="Channel 1")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if imaging_name is not None:
        return BaseROIExtractorImaging(
            imaging_name=imaging_name, file_path=str(file_path), **kwargs
        )

    suffix = file_path.suffix.lower()

    extractor_priority = []

    if suffix in [".tif", ".tiff"]:
        extractor_priority = [
            "ScanImageImagingExtractor",
            "BrukerTiffSinglePlaneImagingExtractor",
            "TiffImagingExtractor",  # Generic fallback
        ]
    elif suffix == ".sbx":
        extractor_priority = ["SbxImagingExtractor"]
    elif suffix in [".h5", ".hdf5"]:
        extractor_priority = ["Hdf5ImagingExtractor"]
    elif suffix == ".nwb":
        extractor_priority = ["NwbImagingExtractor"]
    elif suffix == ".npy":
        extractor_priority = ["NumpyImagingExtractor"]
    else:
        raise ValueError(f"No suitable imaging extractor found for {file_path}. ")

    last_error = None
    for extractor_name in extractor_priority:
        if extractor_name not in imaging_extractor_dict:
            continue

        try:
            imaging = BaseROIExtractorImaging(
                imaging_name=extractor_name, file_path=str(file_path), **kwargs
            )
            print(f"Successfully loaded with {extractor_name}")
            return imaging
        except Exception as e:
            last_error = e
            continue

    # If all attempts failed, raise the last error
    if last_error:
        raise RuntimeError(
            f"Could not load imaging data from {file_path}. "
            f"Tried extractors: {extractor_priority}. "
            f"Last error: {last_error}"
        )
    else:
        raise RuntimeError(
            f"No suitable imaging extractor found for {file_path}. "
            f"Supported formats: .tif, .tiff, .sbx, .h5, .hdf5, .nwb"
        )
