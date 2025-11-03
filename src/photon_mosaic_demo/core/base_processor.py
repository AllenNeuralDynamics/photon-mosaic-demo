"""Base class for all processing modules in the photon-mosaic-demo repository."""
import logging
from abc import ABC, abstractmethod
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional


class BaseProcessor(ABC):
    """Abstract base class for all processing modules.
    
    This class provides a common interface for all processing modules
    in the photon-mosaic-demo repository. Each processor should inherit
    from this class and implement the run method.
    """
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """Initialize the base processor.
        
        Parameters
        ----------
        input_dir : Path
            Input directory containing data to process
        output_dir : Path
            Output directory for processed results
        logger : Optional[logging.Logger]
            Logger instance. If None, creates a default logger
        **kwargs
            Additional keyword arguments specific to the processor
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.start_time = dt.now()
        self.end_time: Optional[dt] = None
        
        # Set up logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
            
        # Store additional parameters
        self.config = kwargs
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute the processing pipeline.
        
        This method should be implemented by each processor to perform
        the main processing logic.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing processing results and metadata
        """
        pass
    
    def _finalize(self) -> None:
        """Finalize processing by recording end time."""
        self.end_time = dt.now()
        duration = self.end_time - self.start_time
        self.logger.info(f"Processing completed in {duration}")
    
    def get_processing_metadata(self) -> Dict[str, Any]:
        """Get metadata about the processing run.
        
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary containing timing and configuration info
        """
        return {
            "processor_name": self.__class__.__name__,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "config": self.config
        }