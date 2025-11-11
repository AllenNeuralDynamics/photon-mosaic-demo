"""Map ROI extractor implementations to BaseImaging and BaseRois"""
# TODO
from ..core import BaseImaging, BaseImagingSegment

from roiextractors.imagingextractor import ImagingExtractor
from roiextractors.extractorlist import imaging_extractor_dict


class BaseROIExtractorImaging(BaseImaging):
    """Base class for ROI extractors that work with BaseImaging data."""

    def __init__(self, imaging_name: str, **kwargs):
        self.roiextractor_imaging_class = imaging_extractor_dict[imaging_name]

        roi_extractor = self.roiextractor_imaging_class(**kwargs)
        width, height = roi_extractor.get_sample_shape()
        
        segment = BaseROIExtractorImagingSegment(roi_extractor)
        BaseImaging.__init__(self, shape=(width, height), sampling_frequency=roi_extractor.get_sampling_frequency())
        self.add_imaging_segment(segment)
        self.name = f"{imaging_name} (ROIExtractors)"

        self._kwargs = {
            "imaging_name": imaging_name,
            **kwargs
        }
        
  
class BaseROIExtractorImagingSegment(BaseImagingSegment):
    """Base class for ROI extractors that work with BaseImaging data."""

    def __init__(self, roi_extractor_imaging: ImagingExtractor):
        BaseImagingSegment.__init__(self, sampling_frequency=roi_extractor_imaging.get_sampling_frequency())
        self.roiextractor_extractor = roi_extractor_imaging
        
    def get_num_samples(self):
        return self.roiextractor_extractor.get_num_samples()

    def get_series(self, start_frame = None, end_frame = None):
        return self.roiextractor_extractor.get_series(start_frame, end_frame)


def auto_add_roiextractor_methods():
    pass
    # # TODO something like this to auto add all file formats of the roiextractor to the BaseROIExtractorImagingSegment
    # for imaging_name, imagin_class in imaging_extractor_dict.items():
    #     pass

    #     # class ...
    #     # setattr(
    #     #     BaseROIExtractorImagingSegment,
    #     method_name,
    #     lambda self, *args, method_name=method_name, **kwargs: getattr(self.roiextractor_extractor, method_name)(*args, **kwargs)
    # )