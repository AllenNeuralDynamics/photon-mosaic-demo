"""Map ROI extractor implementations to BaseImaging and BaseRois"""
# TODO
from ..core import BaseImaging, BaseRois

class BaseROIExtractorImaging(BaseImaging):
    """Base class for ROI extractors that work with BaseImaging data."""

    def __init__(self, imaging: BaseImaging):
        super().__init__(imaging)


class BaseROIExtractorRois(BaseRois):
    """Base class for ROI extractors that work with BaseRois data."""

    def __init__(self, rois: BaseRois):
        super().__init__(rois)