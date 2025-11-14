from math import prod

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from roiextractors.core_utils import _convert_bytes_to_str, _convert_seconds_to_str
from spikeinterface.core.base import BaseExtractor, BaseSegment


class BaseRois(BaseExtractor):
    """Base class for rois extractors."""

    def __init__(
        self,
        sampling_frequency: float,
        shape: tuple | list | np.ndarray,
        roi_ids: ArrayLike,
    ):
        BaseExtractor.__init__(self, roi_ids)
        self._sampling_frequency = float(sampling_frequency)
        assert (
            len(shape) == 2
        ), "Shape must be a tuple/list/array of length 2 (width, height)"
        self._image_shape = np.array(shape)

        # no concept of segments for rois yet, since they are spatial only

    # def __repr__(self):
    #     return self._repr_header()

    # def _repr_header(self, display_name=True):
    #     """Generate text representation of the BaseImaging object."""
    #     num_samples = [self.get_num_samples(segment_index=i) for i in range(self.get_num_segments())]
    #     sample_shape = self.get_sample_shape()
    #     dtype = self.get_dtype()
    #     sf_hz = self.sampling_frequency

    #     # Format sampling frequency
    #     if not sf_hz.is_integer():
    #         sampling_frequency_repr = f"{sf_hz:f} Hz"
    #     else:
    #         sampling_frequency_repr = f"{sf_hz:0.1f}Hz"

    #     # Calculate duration
    #     duration = num_samples / sf_hz
    #     duration_repr = _convert_seconds_to_str(duration)

    #     # Calculate memory size using product of all dimensions in image_size
    #     memory_size = num_samples * prod(sample_shape) * dtype.itemsize
    #     memory_repr = _convert_bytes_to_str(memory_size)

    #     # Format shape string based on whether data is volumetric or not
    #     sample_shape_repr = f"{sample_shape[0]} rows x {sample_shape[1]} columns "
    #     if self.is_volumetric:
    #         sample_shape_repr += f"x {sample_shape[2]} planes"

    #     return (
    #         f"{self.name}\n"
    #         f"  Number of segments: {self.get_num_segments()} \n"
    #         f"  Sample shape: {sample_shape_repr} \n"
    #         f"  Number of samples: {num_samples:,} \n"
    #         f"  Sampling rate: {sampling_frequency_repr}\n"
    #         f"  Duration: {duration_repr}\n"
    #         f"  Imaging data memory: {memory_repr} ({dtype} dtype)"
    #     )

    # def __repr__(self):
    #     return self._repr_text()

    # def _repr_html_(self, display_name=True):
    #     common_style = "margin-left: 10px;"
    #     border_style = "border:1px solid #ddd; padding:10px;"

    #     html_header = f"<div style='{border_style}'><strong>{self._repr_header(display_name)}</strong></div>"

    #     html_unit_ids = f"<details style='{common_style}'>  <summary><strong>Unit IDs</strong></summary><ul>"
    #     html_unit_ids += f"{self.unit_ids} </details>"

    #     html_extra = self._get_common_repr_html(common_style)

    #     html_repr = html_header + html_unit_ids + html_extra
    #     return html_repr

    @property
    def image_shape(self):
        """Get the shape of the images (height, width).

        Returns
        -------
        tuple
            The shape of the images as (height, width).
        """
        return self._image_shape

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    @property
    def roi_ids(self) -> ArrayLike:
        """Get the ROI IDs.

        Returns
        -------
        ArrayLike
            The ROI IDs.
        """
        return self._roi_ids

    def get_num_rois(self) -> int:
        """Get the total number of ROIs.

        Returns
        -------
        int
            The total number of ROIs.
        """
        return len(self.roi_ids)

    def get_roi_image(self, roi_id: int) -> np.ndarray:
        """Get the image mask for a specific ROI.

        Parameters
        ----------
        roi_id : int
            The ID of the ROI.

        Returns
        -------
        np.ndarray
            The image mask for the specified ROI.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


# class BaseImagingSegment:
#     """Base class for imaging segments."""

#     def __init__(self, parent_imaging: BaseImaging):
#         self._parent_imaging = parent_imaging


#     def get_num_samples(self) -> int:
#         """Get the total number of samples (frames) in this imaging segment.

#         Returns
#         -------
#         int
#             The total number of samples (frames).
#         """
#         raise NotImplementedError("This method should be implemented in subclasses.")


#     def get_series(start_frame: int, end_frame: int) -> np.ndarray:
#         """Get a series of frames from the imaging segment.

#         Parameters
#         ----------
#         start_frame : int
#             The starting frame index (inclusive).
#         end_frame : int
#             The ending frame index (exclusive).

#         Returns
#         -------
#         np.ndarray
#             The requested series of frames as a NumPy array.
#         """
#         raise NotImplementedError("This method should be implemented in subclasses.")


#     def get_times(self) -> np.ndarray:
#         """Get the timestamps for each frame in this imaging segment.

#         Returns
#         -------
#         np.ndarray
#             The timestamps for each frame.
#         """
#         raise NotImplementedError("This method should be implemented in subclasses.")


class BaseImagingSegment(BaseSegment):
    """
    Abstract class representing a multichannel timeseries, or block of raw ephys traces
    """

    def __init__(self, sampling_frequency=None, t_start=None, time_vector=None):
        # sampling_frequency and time_vector are exclusive
        if sampling_frequency is None:
            assert (
                time_vector is not None
            ), "Pass either 'sampling_frequency' or 'time_vector'"
            assert time_vector.ndim == 1, "time_vector should be a 1D array"

        if time_vector is None:
            assert (
                sampling_frequency is not None
            ), "Pass either 'sampling_frequency' or 'time_vector'"

        self.sampling_frequency = sampling_frequency
        self.t_start = t_start
        self.time_vector = time_vector

        BaseSegment.__init__(self)

    def get_times(self) -> np.ndarray:
        if self.time_vector is not None:
            self.time_vector = np.asarray(self.time_vector)
            return self.time_vector
        else:
            time_vector = np.arange(self.get_num_samples(), dtype="float64")
            time_vector /= self.sampling_frequency
            if self.t_start is not None:
                time_vector += self.t_start
            return time_vector

    def get_start_time(self) -> float:
        if self.time_vector is not None:
            return self.time_vector[0]
        else:
            return self.t_start if self.t_start is not None else 0.0

    def get_end_time(self) -> float:
        if self.time_vector is not None:
            return self.time_vector[-1]
        else:
            t_stop = (self.get_num_samples() - 1) / self.sampling_frequency
            if self.t_start is not None:
                t_stop += self.t_start
            return t_stop

    def get_times_kwargs(self) -> dict:
        """
        Retrieves the timing attributes characterizing a RecordingSegment

        Returns
        -------
        dict
            A dictionary containing the following key-value pairs:

            - "sampling_frequency" : The sampling frequency of the RecordingSegment.
            - "t_start" : The start time of the RecordingSegment.
            - "time_vector" : The time vector of the RecordingSegment.

        Notes
        -----
        The keys are always present, but the values may be None.
        """
        time_kwargs = dict(
            sampling_frequency=self.sampling_frequency,
            t_start=self.t_start,
            time_vector=self.time_vector,
        )
        return time_kwargs

    def sample_index_to_time(self, sample_ind):
        """
        Transform sample index into time in seconds
        """
        if self.time_vector is None:
            time_s = sample_ind / self.sampling_frequency
            if self.t_start is not None:
                time_s += self.t_start
        else:
            time_s = self.time_vector[sample_ind]
        return time_s

    def time_to_sample_index(self, time_s):
        """
        Transform time in seconds into sample index
        """
        if self.time_vector is None:
            if self.t_start is None:
                sample_index = time_s * self.sampling_frequency
            else:
                sample_index = (time_s - self.t_start) * self.sampling_frequency
            sample_index = np.round(sample_index).astype(np.int64)
        else:
            sample_index = np.searchsorted(self.time_vector, time_s, side="right") - 1

        return sample_index

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal segment

        Returns:
            SampleIndex : Number of samples in the signal segment
        """
        # must be implemented in subclass
        raise NotImplementedError

    def get_series(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> np.ndarray:
        """
        Return the raw series, optionally for a subset of samples

        Parameters
        ----------
        start_frame : int | None, default: None
            start sample index, or zero if None
        end_frame : int | None, default: None
            end_sample, or number of samples if None

        Returns
        -------
        traces : np.ndarray
            Array of traces, num_samples x num_channels
        """
        # must be implemented in subclass
        raise NotImplementedError
