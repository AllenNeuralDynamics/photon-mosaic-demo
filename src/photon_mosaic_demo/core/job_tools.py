from __future__ import annotations
import numpy as np
import platform
import warnings
from tqdm.auto import tqdm

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import threading

from spikeinterface.core.job_tools import (
    ensure_n_jobs, divide_recording_into_chunks, process_worker_initializer, process_function_wrapper,
    thread_worker_initializer, thread_function_wrapper, chunk_duration_to_chunk_size
)
from spikeinterface.core.core_tools import convert_string_to_bytes, convert_bytes_to_str, convert_seconds_to_str



class BaseChunkExecutor:
    """
    Base class for chunk execution.
    """

    def __init__(
        self,
        extractor: "BaseExtractor",
        func,
        init_func,
        init_args,
        verbose=False,
        progress_bar=False,
        handle_returns=False,
        gather_func=None,
        pool_engine="thread",
        n_jobs=1,
        total_memory=None,
        chunk_size=None,
        chunk_memory=None,
        chunk_duration=None,
        mp_context=None,
        job_name="",
        max_threads_per_worker=1,
        need_worker_index=False,
    ):
        self.extractor = extractor
        self.func = func
        self.init_func = init_func
        self.init_args = init_args

        if pool_engine == "process":
            if mp_context is None:
                if hasattr(extractor, "get_preferred_mp_context"):
                    mp_context = extractor.get_preferred_mp_context()
            if mp_context is not None and platform.system() == "Windows":
                assert mp_context != "fork", "'fork' mp_context not supported on Windows!"
            elif mp_context == "fork" and platform.system() == "Darwin":
                warnings.warn('As of Python 3.8 "fork" is no longer considered safe on macOS')

        self.mp_context = mp_context

        self.verbose = verbose
        self.progress_bar = progress_bar

        self.handle_returns = handle_returns
        self.gather_func = gather_func

        self.n_jobs = ensure_n_jobs(self.extractor, n_jobs=n_jobs)
        self.chunk_size = self.ensure_chunk_size(
            total_memory=total_memory,
            chunk_size=chunk_size,
            chunk_memory=chunk_memory,
            chunk_duration=chunk_duration,
            n_jobs=self.n_jobs,
        )
        self.job_name = job_name
        self.max_threads_per_worker = max_threads_per_worker

        self.pool_engine = pool_engine

        self.need_worker_index = need_worker_index

        if verbose:
            chunk_memory = self.get_chunk_memory()
            total_memory = chunk_memory * self.n_jobs
            chunk_duration = self.chunk_size / extractor.sampling_frequency
            chunk_memory_str = convert_bytes_to_str(chunk_memory)
            total_memory_str = convert_bytes_to_str(total_memory)
            chunk_duration_str = convert_seconds_to_str(chunk_duration)
            print(
                self.job_name,
                "\n"
                f"engine={self.pool_engine} - "
                f"n_jobs={self.n_jobs} - "
                f"samples_per_chunk={self.chunk_size:,} - "
                f"chunk_memory={chunk_memory_str} - "
                f"total_memory={total_memory_str} - "
                f"chunk_duration={chunk_duration_str}",
            )

    def get_chunk_memory(self):
        raise NotImplementedError

    def ensure_chunk_size(
        self, total_memory=None, chunk_size=None, chunk_memory=None, chunk_duration=None, n_jobs=1, **other_kwargs
    ):
        raise NotImplementedError

    def run(self, slices=None):
        """
        Runs the defined jobs.
        """

        if slices is None:
            # TODO: rename
            slices = divide_recording_into_chunks(self.extractor, self.chunk_size)

        if self.handle_returns:
            returns = []
        else:
            returns = None

        if self.n_jobs == 1:
            if self.progress_bar:
                slices = tqdm(slices, desc=f"{self.job_name} (no parallelization)", total=len(slices))

            worker_dict = self.init_func(*self.init_args)
            if self.need_worker_index:
                worker_dict["worker_index"] = 0

            for segment_index, frame_start, frame_stop in slices:
                res = self.func(segment_index, frame_start, frame_stop, worker_dict)
                if self.handle_returns:
                    returns.append(res)
                if self.gather_func is not None:
                    self.gather_func(res)

        else:
            n_jobs = min(self.n_jobs, len(slices))

            if self.pool_engine == "process":

                if self.need_worker_index:
                    lock = multiprocessing.Lock()
                    array_pid = multiprocessing.Array("i", n_jobs)
                    for i in range(n_jobs):
                        array_pid[i] = -1
                else:
                    lock = None
                    array_pid = None

                # parallel
                with ProcessPoolExecutor(
                    max_workers=n_jobs,
                    initializer=process_worker_initializer,
                    mp_context=multiprocessing.get_context(self.mp_context),
                    initargs=(
                        self.func,
                        self.init_func,
                        self.init_args,
                        self.max_threads_per_worker,
                        self.need_worker_index,
                        lock,
                        array_pid,
                    ),
                ) as executor:
                    results = executor.map(process_function_wrapper, slices)

                    if self.progress_bar:
                        results = tqdm(
                            results, desc=f"{self.job_name} (workers: {n_jobs} processes)", total=len(slices)
                        )

                    for res in results:
                        if self.handle_returns:
                            returns.append(res)
                        if self.gather_func is not None:
                            self.gather_func(res)

            elif self.pool_engine == "thread":
                # this is need to create a per worker local dict where the initializer will push the func wrapper
                thread_local_data = threading.local()

                global _thread_started
                _thread_started = 0

                if self.progress_bar:
                    # here the tqdm threading do not work (maybe collision) so we need to create a pbar
                    # before thread spawning
                    pbar = tqdm(desc=f"{self.job_name} (workers: {n_jobs} threads)", total=len(slices))

                if self.need_worker_index:
                    lock = threading.Lock()
                else:
                    lock = None

                with ThreadPoolExecutor(
                    max_workers=n_jobs,
                    initializer=thread_worker_initializer,
                    initargs=(
                        self.func,
                        self.init_func,
                        self.init_args,
                        self.max_threads_per_worker,
                        thread_local_data,
                        self.need_worker_index,
                        lock,
                    ),
                ) as executor:

                    slices2 = [(thread_local_data,) + tuple(args) for args in slices]
                    results = executor.map(thread_function_wrapper, slices2)

                    for res in results:
                        if self.progress_bar:
                            pbar.update(1)
                        if self.handle_returns:
                            returns.append(res)
                        if self.gather_func is not None:
                            self.gather_func(res)
                if self.progress_bar:
                    pbar.close()
                    del pbar

            else:
                raise ValueError("If n_jobs>1 pool_engine must be 'process' or 'thread'")

        return returns


class ChunkImagingExecutor(BaseChunkExecutor):
    """
    Core class for parallel processing to run a "function" over chunks on a recording.

    It supports running a function:
        * in loop with chunk processing (low RAM usage)
        * at once if chunk_size is None (high RAM usage)
        * in parallel with ProcessPoolExecutor (higher speed)

    The initializer ("init_func") allows to set a global context to avoid heavy serialization
    (for examples, see implementation in `core.waveform_tools`).

    Parameters
    ----------
    imaging : RecordBaseImagingngExtractor
        The imaging to be processed
    func : function
        Function that runs on each chunk
    init_func : function
        Initializer function to set the global context (accessible by "func")
    init_args : tuple
        Arguments for init_func
    verbose : bool
        If True, output is verbose
    job_name : str, default: ""
        Job name
    progress_bar : bool, default: False
        If True, a progress bar is printed to monitor the progress of the process
    handle_returns : bool, default: False
        If True, the function can return values
    gather_func : None or callable, default: None
        Optional function that is called in the main thread and retrieves the results of each worker.
        This function can be used instead of `handle_returns` to implement custom storage on-the-fly.
    pool_engine : "process" | "thread", default: "thread"
        If n_jobs>1 then use ProcessPoolExecutor or ThreadPoolExecutor
    n_jobs : int, default: 1
        Number of jobs to be used. Use -1 to use as many jobs as number of cores
    total_memory : str, default: None
        Total memory (RAM) to use (e.g. "1G", "500M")
    chunk_memory : str, default: None
        Memory per chunk (RAM) to use (e.g. "1G", "500M")
    chunk_size : int or None, default: None
        Size of each chunk in number of samples. If "total_memory" or "chunk_memory" are used, it is ignored.
    chunk_duration : str or float or None
        Chunk duration in s if float or with units if str (e.g. "1s", "500ms")
    mp_context : "fork" | "spawn" | None, default: None
        "fork" or "spawn". If None, the context is taken by the recording.get_preferred_mp_context().
        "fork" is only safely available on LINUX systems.
    max_threads_per_worker : int or None, default: None
        Limit the number of thread per process using threadpoolctl modules.
        This used only when n_jobs>1
        If None, no limits.
    need_worker_index : bool, default False
        If True then each worker will also have a "worker_index" injected in the local worker dict.

    Returns
    -------
    res : list
        If "handle_returns" is True, the results for each chunk process
    """

    def __init__(
        self,
        imaging,
        func,
        init_func,
        init_args,
        verbose=False,
        progress_bar=False,
        handle_returns=False,
        gather_func=None,
        pool_engine="thread",
        n_jobs=1,
        total_memory=None,
        chunk_size=None,
        chunk_memory=None,
        chunk_duration=None,
        mp_context=None,
        job_name="",
        max_threads_per_worker=1,
        need_worker_index=False,
    ):
        self.imaging = imaging
        super().__init__(
            imaging,
            func,
            init_func,
            init_args,
            verbose=verbose,
            progress_bar=progress_bar,
            handle_returns=handle_returns,
            gather_func=gather_func,
            pool_engine=pool_engine,
            n_jobs=n_jobs,
            total_memory=total_memory,
            chunk_size=chunk_size,
            chunk_memory=chunk_memory,
            chunk_duration=chunk_duration,
            mp_context=mp_context,
            job_name=job_name,
            max_threads_per_worker=max_threads_per_worker,
            need_worker_index=need_worker_index,
        )

    def get_chunk_memory(self):
        image_shape = self.imaging.image_shape
        return self.chunk_size * self.imaging.get_dtype().itemsize * image_shape[0] * image_shape[1]

    def ensure_chunk_size(
        self, total_memory=None, chunk_size=None, chunk_memory=None, chunk_duration=None, n_jobs=1, **other_kwargs
    ):
        """
        "chunk_size" is the traces.shape[0] for each worker.

        Flexible chunk_size setter with 3 ways:
            * "chunk_size" : is the length in sample for each chunk independently of channel count and dtype.
            * "chunk_memory" : total memory per chunk per worker
            * "total_memory" : total memory over all workers.

        If chunk_size/chunk_memory/total_memory are all None then there is no chunk computing
        and the full trace is retrieved at once.

        Parameters
        ----------
        chunk_size : int or None
            size for one chunk per job
        chunk_memory : str or None
            must end with "k", "M", "G", etc for decimal units and "ki", "Mi", "Gi", etc for
            binary units. (e.g. "1k", "500M", "2G", "1ki", "500Mi", "2Gi")
        total_memory : str or None
            must end with "k", "M", "G", etc for decimal units and "ki", "Mi", "Gi", etc for
            binary units. (e.g. "1k", "500M", "2G", "1ki", "500Mi", "2Gi")
        chunk_duration : None or float or str
            Units are second if float.
            If str then the str must contain units(e.g. "1s", "500ms")
        """
        if chunk_size is not None:
            # manual setting
            chunk_size = int(chunk_size)
        elif chunk_memory is not None:
            assert total_memory is None
            # set by memory per worker size
            chunk_memory = convert_string_to_bytes(chunk_memory)
            n_bytes = np.dtype(self.imaging.get_dtype()).itemsize
            n_pixels = self.imaging.get_num_pixels()
            chunk_size = int(chunk_memory / (n_pixels * n_bytes))
        elif total_memory is not None:
            # clip by total memory size
            n_jobs = ensure_n_jobs(self.imaging, n_jobs=n_jobs)
            total_memory = convert_string_to_bytes(total_memory)
            n_bytes = np.dtype(self.imaging.get_dtype()).itemsize
            n_pixels = self.imaging.get_num_pixels()
            chunk_size = int(total_memory / (n_pixels * n_bytes * n_jobs))
        elif chunk_duration is not None:
            chunk_size = chunk_duration_to_chunk_size(chunk_duration, self.imaging)
        else:
            # Edge case to define single chunk per segment for n_jobs=1.
            # All chunking parameters equal None mean single chunk per segment
            if n_jobs == 1:
                num_segments = self.imaging.get_num_segments()
                samples_in_larger_segment = max([self.imaging.get_num_samples(segment) for segment in range(num_segments)])
                chunk_size = samples_in_larger_segment
            else:
                raise ValueError("For n_jobs >1 you must specify total_memory or chunk_size or chunk_memory")
        return chunk_size