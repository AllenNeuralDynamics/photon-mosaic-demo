from __future__ import annotations

import numpy as np
from spikeinterface.core.core_tools import (
    convert_bytes_to_str,
    convert_seconds_to_str,
    convert_string_to_bytes,
)
from spikeinterface.core.job_tools import (
    ChunkRecordingExecutor,
    chunk_duration_to_chunk_size,
    ensure_n_jobs,
)
