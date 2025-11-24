import re
from roiextractors.extractorlist import imaging_extractor_dict

# Import base classes
from photon_mosaic_demo.extractors.baseroiextractors import (
    BaseROIExtractorImaging,
    BaseROIExtractorImagingSegment,
    get_imaging_extractor,
)

# Build __all__ to include all exports
__all__ = [
    "BaseROIExtractorImaging",
    "BaseROIExtractorImagingSegment",
    "get_imaging_extractor",
]

# Dynamically create classes and read functions for all imaging extractors
for imaging_name, imaging_class in imaging_extractor_dict.items():
    # Dynamically create a class for each imaging extractor
    # The class inherits from BaseROIExtractorImaging and sets imaging_name automatically

    def make_class(extractor_name):
        """Factory function to create a class with proper closure over extractor_name."""

        class DynamicImagingClass(BaseROIExtractorImaging):
            """Dynamically generated imaging class for {extractor_name}."""

            def __init__(self, **kwargs):
                super().__init__(imaging_name=extractor_name, **kwargs)

        # Set proper class name and module
        DynamicImagingClass.__name__ = extractor_name
        DynamicImagingClass.__qualname__ = extractor_name
        DynamicImagingClass.__doc__ = f"""Adapter class for {extractor_name}.
        
        This class wraps the {extractor_name} from roiextractors to work with BaseImaging.
        
        Parameters
        ----------
        **kwargs
            Keyword arguments passed to {extractor_name}.
        
        Examples
        --------
        >>> imaging = {extractor_name}(file_path="path/to/data.tif")
        >>> frames = imaging.get_series(start_frame=0, end_frame=10)
        """

        return DynamicImagingClass

    # Create the class
    dynamic_class = make_class(imaging_name)

    # Add the class to the current module
    globals()[imaging_name] = dynamic_class
    __all__.append(imaging_name)

    # Create a read_* function that instantiates the class
    def make_read_function(cls, extractor_cls):
        """Factory function to create a read function with proper closure and signature."""
        import inspect
        from functools import wraps

        # Get the signature from the original extractor class
        try:
            sig = inspect.signature(extractor_cls.__init__)
            # Remove 'self' parameter
            params = [p for name, p in sig.parameters.items() if name != "self"]
        except Exception:
            # Fallback if we can't get the signature
            params = [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)]

        def read_function(*args, **kwargs):
            """Convenience function to create an instance of {cls.__name__}.

            Returns
            -------
            {cls.__name__}
                An instance of the imaging class.
            """
            return cls(*args, **kwargs)

        # Set the signature on the function
        read_function.__signature__ = inspect.Signature(parameters=params)

        return read_function

    # Create and add the read function
    # Convert CamelCase to snake_case, handling consecutive capitals like TIFF
    # Insert underscore before uppercase letters that are followed by lowercase
    snake_case_name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", imaging_name)
    snake_case_name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", snake_case_name)
    snake_case_name = snake_case_name.lower()
    # Remove _extractor suffix
    snake_case_name = snake_case_name.replace("_extractor", "").replace("_imaging", "")
    read_func_name = f"read_{snake_case_name}"
    read_func = make_read_function(dynamic_class, imaging_class)
    read_func.__name__ = read_func_name
    read_func.__qualname__ = read_func_name
    read_func.__doc__ = f"""Convenience function to create an instance of {imaging_name}.
    
    Returns
    -------
    {imaging_name}
        An instance of the imaging class.
    
    Examples
    --------
    >>> imaging = {read_func_name}(file_path="path/to/data.tif")
    >>> frames = imaging.get_series(start_frame=0, end_frame=10)
    """

    globals()[read_func_name] = read_func
    __all__.append(read_func_name)
