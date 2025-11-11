import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from math import prod


from roiextractors.core_utils import _convert_bytes_to_str, _convert_seconds_to_str
from spikeinterface.core.base import BaseExtractor, BaseSegment


class BaseImaging(BaseExtractor):
    """Base class for imaging extractors."""

    def __init__(self, sampling_frequency: float, shape: tuple | list | np.ndarray, channel_ids: list | None = None):
        if channel_ids is None:
            channel_ids = [0]  # fake channel
        BaseExtractor.__init__(self, channel_ids)
        self._sampling_frequency = float(sampling_frequency)
        assert len(shape) == 2, "Shape must be a tuple/list/array of length 2 (width, height)"
        self._image_shape = np.array(shape)
        self._imaging_segments: list[BaseImagingSegment] = []

    def __repr__(self):
        return self._repr_header()

    def _repr_text(self, display_name=True):
        """Generate text representation of the BaseImaging object."""
        num_samples = [self.get_num_samples(segment_index=i) for i in range(self.get_num_segments())]
        image_shape = self.image_shape
        dtype = self.get_dtype()
        sf_hz = self.sampling_frequency

        # Format sampling frequency
        if not sf_hz.is_integer():
            sampling_frequency_repr = f"{sf_hz:f} Hz"
        else:
            sampling_frequency_repr = f"{sf_hz:0.1f}Hz"

        # Calculate duration
        durations = [ns / sf_hz for ns in num_samples]
        duration_repr = [_convert_seconds_to_str(duration) for duration in durations]

        # Calculate memory size using product of all dimensions in image_size
        memory_sizes = [ns * prod(image_shape) * dtype.itemsize for ns in num_samples]
        memory_repr = [_convert_bytes_to_str(memory_size) for memory_size in memory_sizes]

        if self.get_num_segments() == 1:
            num_samples = num_samples[0]
            duration_repr = duration_repr[0]
            memory_repr = memory_repr[0]

        # Format shape string based on whether data is volumetric or not
        image_shape_repr = f"{image_shape[0]} rows x {image_shape[1]} columns "
        return (
            f"{self.name}\n"
            f"  Number of segments: {self.get_num_segments()} \n"
            f"  Sample shape: {image_shape_repr} \n"
            f"  Number of samples: {num_samples:,} \n"
            f"  Sampling rate: {sampling_frequency_repr}\n"
            f"  Duration: {duration_repr}\n"
            f"  Imaging data memory: {memory_repr} ({dtype} dtype)"
        )

    def __repr__(self):
        return self._repr_text()

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


    def get_num_samples(self, segment_index: int | None = None) -> int:
        """Get the total number of samples (frames) in the imaging data.

        Parameters
        ----------

        Returns
        -------
        int
            The total number of samples (frames).
        """
        if segment_index is None:
            if self.get_num_segments() == 1:
                segment_index = 0
            else:
                raise ValueError("segment_index must be provided for multi-segment imaging data.")
        return self._imaging_segments[segment_index].get_num_samples()

    def get_num_segments(self) -> int:
        return len(self._imaging_segments)


    def get_dtype(self) -> DTypeLike:
        """Get the data type of the video.

        Returns
        -------
        dtype: dtype
            Data type of the video.
        """
        return self.get_series(start_frame=0, end_frame=2, segment_index=0).dtype

    def get_series(self, start_frame: int | None = None, end_frame: int | None = None, segment_index: int | None = None) -> np.ndarray:
        """Get a series of frames from the imaging data.

        Parameters
        ----------
        start_sample : int
            The starting frame index (inclusive).
        end_sample : int
            The ending frame index (exclusive).
        segment_index : int | None
            The index of the imaging segment. If None and there is only one segment, it defaults to 0.

        Returns
        -------
        np.ndarray
            The requested series of frames as a NumPy array.
        """
        if segment_index is None:
            if self.get_num_segments() == 1:
                segment_index = 0
            else:
                raise ValueError("segment_index must be provided for multi-segment imaging data.")
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else self.get_num_samples(segment_index=segment_index)
        return self._imaging_segments[segment_index].get_series(start_frame, end_frame)

    def add_imaging_segment(self, imaging_segment):
        """Adds an imaging segment.

        Parameters
        ----------
        imaging_segment : BaseImagingSegment
            The imaging segment to add.
        """
        self._imaging_segments.append(imaging_segment)
        imaging_segment.set_parent_extractor(self)

    def get_times(self, segment_index: int | None = None) -> np.ndarray:
        """Get the timestamps for each frame in the imaging data.

        Parameters
        ----------
        segment_index : int | None
            The index of the imaging segment. If None and there is only one segment, it defaults to 0.

        Returns
        -------
        np.ndarray
            The timestamps for each frame.
        """
        if segment_index is None:
            if self.get_num_segments() == 1:
                segment_index = 0
            else:
                raise ValueError("segment_index must be provided for multi-segment imaging data.")
        return self._imaging_segments[segment_index].get_times()

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
            assert time_vector is not None, "Pass either 'sampling_frequency' or 'time_vector'"
            assert time_vector.ndim == 1, "time_vector should be a 1D array"

        if time_vector is None:
            assert sampling_frequency is not None, "Pass either 'sampling_frequency' or 'time_vector'"

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
            sampling_frequency=self.sampling_frequency, t_start=self.t_start, time_vector=self.time_vector
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
