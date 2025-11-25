from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class EdgeDetectorImaging(BasePreprocessor):

    def __init__(self, imaging, **kwargs):

        BasePreprocessor.__init__(self, imaging)
        for parent_segment in imaging._imaging_segments:
            segment = EdgeDetectorImagingSegment(parent_segment)
            self.add_imaging_segment(segment)

        self._kwargs = dict(imaging=imaging)


class EdgeDetectorImagingSegment(BasePreprocessorSegment):

    def __init__(self, parent_imaging_segment):
        BasePreprocessorSegment.__init__(self, parent_imaging_segment)

    def get_series(self, start_frame, end_frame):
        import numpy as np
        from scipy.ndimage import sobel

        data = self.parent_imaging_segment.get_series(start_frame, end_frame)
        filtered_data = sobel(data, axis=-1)
        return filtered_data


edge_detector = EdgeDetectorImaging
