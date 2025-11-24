from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path


class MotionCorrectionSettings(BaseSettings, cli_parse_args=True):
    """Settings for Suite2P motion correction.

    This class defines all configuration parameters for motion correction using Suite2P.
    Values can be provided via constructor, environment variables, or .env file.
    """

    debug: bool = Field(default=False, description="Run with partial dataset")
    tmp_dir: Path = Field(
        default=Path("/scratch"),
        description="Directory into which to write temporary files produced by Suite2P",
    )
    data_type: str = Field(default="h5", description="Processing h5 (default) or TIFF timeseries")
    do_registration: bool = Field(
        default="true",
        description="whether to register data (2 forces re-registration)",
    )
    batch_size: int = Field(default=500, description="Number of frames per batch")
    align_by_chan: int = Field(
        default=1,
        description="when multi-channel, you can align by non-functional channel " "(1-based)",
    )
    maxregshift: float = Field(
        default=0.1,
        description="max allowed registration shift, as a fraction of "
        "frame max(width and height). This will be ignored if force_refImg is set to True",
    )
    force_refImg: bool = Field(default=True, description="Force the use of an external reference image")
    nonrigid: bool = Field(default=True, description="Whether to use non-rigid registration")
    block_size: list = Field(default_factory=lambda: [128, 128], description="Block size for non-rigid registration.")
    snr_thresh: float = Field(
        default=1.2,
        description="if any nonrigid block is below this threshold, it gets smoothed "
        "until above this threshold. 1.0 results in no smoothing",
    )
    maxregshiftNR: int = Field(
        default=5,
        description="maximum pixel shift allowed for nonrigid, relative to rigid",
    )
    outlier_detrend_window: float = Field(
        default=3.0,
        description="For outlier rejection in the xoff/yoff outputs of suite2p, "
        "the offsets are first de-trended with a median filter of this duration [seconds]. "
        "This value is ~30 or 90 samples in size for 11 and 31 Hz sampling rates respectively.",
    )
    outlier_maxregshift: float = Field(
        default=0.05,
        description="Units [fraction FOV dim]. After median-filter detrending, outliers more "
        "than this value are clipped to this value in x and y offset, independently. "
        "This is similar to Suite2P's internal maxregshift, but allows for low-frequency drift. "
        "Default value of 0.05 is typically clipping outliers to 512 * 0.05 = 25 "
        "pixels above or below the median trend.",
    )
    clip_negative: bool = Field(
        default=False,
        description="Whether or not to clip negative pixel values in output. Because the pixel values "
        "in the raw movies are set by the current coming off a photomultiplier tube, there can "
        "be pixels with negative values (current has a sign), possibly due to noise in the rig. "
        "Some segmentation algorithms cannot handle negative values in the movie, so we have this "
        "option to artificially set those pixels to zero.",
    )
    max_reference_iterations: int = Field(
        default=8,
        description="Maximum number of iterations for creating a reference image",
    )
    auto_remove_empty_frames: bool = Field(
        default=True,
        description="Automatically detect empty noise frames at the start and end of the movie. "
        "Overrides values set in trim_frames_start and trim_frames_end. Some movies arrive with "
        "otherwise quality data but contain a set of frames that are empty and contain pure noise. "
        "When processed, these frames tend to receive large random shifts that throw off motion border "
        "calculation. Turning on this setting automatically detects these frames before processing and "
        "removes them from reference image creation, automated smoothing parameter searches, and "
        "finally the motion border calculation. The frames are still written however any "
        "shift estimated is removed and their shift is set to 0 to avoid large motion borders.",
    )
    trim_frames_start: int = Field(
        default=0,
        description="Number of frames to remove from the start of the movie if known. "
        "Removes frames from motion border calculation and resets the frame shifts found"
        "Frames are still written to motion correction. Raises an error if "
        "auto_remove_empty_frames is set and trim_frames_start > 0",
    )
    trim_frames_end: int = Field(
        default=0,
        description="Number of frames to remove from the end of the movie if known. "
        "Removes frames from motion border calculation and resets the frame shifts "
        "found. Frames are still written to motion correction. Raises an error if "
        "auto_remove_empty_frames is set and trim_frames_start > 0",
    )
    do_optimize_motion_params: bool = Field(
        default=False,
        description="Do a search for best parameters of smooth_sigma and smooth_sigma_time. "
        "Adds significant runtime cost to motion correction and should only be run once per "
        "experiment with the resulting parameters being stored for later use.",
    )
    smooth_sigma_time: int = Field(
        default=0,
        description="gaussian smoothing in time. If do_optimize_motion_params is set, " "this will be overridden",
    )
    smooth_sigma: float = Field(
        default=1.15,
        description="~1 good for 2P recordings, recommend 3-5 for 1P recordings. "
        "If do_optimize_motion_params is set, this will be overridden",
    )
    use_ave_image_as_reference: bool = Field(
        default=False,
        description="Only available if `do_optimize_motion_params` is set. "
        "After the a best set of smoothing parameters is found, "
        "use the resulting average image as the reference for the full registration. "
        "This can be used as two step registration by setting by setting "
        "smooth_sigma_min=smooth_sigma_max and smooth_sigma_time_min=smooth_sigma_time_max and steps=1.",
    )
    # Additional parameters that were hardcoded in the original code
    movie_lower_quantile: float = Field(
        default=0.1,
        description="Lower quantile threshold for avg projection histogram adjustment of movie",
    )
    movie_upper_quantile: float = Field(
        default=0.999,
        description="Upper quantile threshold for avg projection histogram adjustment of movie",
    )
    preview_frame_bin_seconds: float = Field(
        default=2.0,
        description="Before creating the webm, the movies will be averaged into bins of this many seconds",
    )
    preview_playback_factor: float = Field(
        default=10.0,
        description="The preview movie will playback at this factor times real-time",
    )
    n_batches: int = Field(
        default=20,
        description="Number of batches to load from the movie for smoothing parameter testing. "
        "Batches are evenly spaced throughout the movie.",
    )
    smooth_sigma_min: float = Field(
        default=0.65,
        description="Minimum value of the parameter search for smooth_sigma",
    )
    smooth_sigma_max: float = Field(
        default=2.15,
        description="Maximum value of the parameter search for smooth_sigma",
    )
    smooth_sigma_steps: int = Field(
        default=4,
        description="Number of steps to grid between smooth_sigma and smooth_sigma_max",
    )
    smooth_sigma_time_min: float = Field(
        default=0,
        description="Minimum value of the parameter search for smooth_sigma_time",
    )
    smooth_sigma_time_max: float = Field(
        default=6,
        description="Maximum value of the parameter search for smooth_sigma_time",
    )
    smooth_sigma_time_steps: int = Field(
        default=7,
        description="Number of steps to grid between smooth_sigma and smooth_sigma_time_max. "
        "Large values will add significant time to motion correction",
    )

    class Config:
        env_prefix = "MOTION_CORRECTION_"
        case_sensitive = False
        env_file = ".env"
