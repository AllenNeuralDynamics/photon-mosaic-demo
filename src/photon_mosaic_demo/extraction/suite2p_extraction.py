from suite2p import io, pipeline, default_ops

def suite2p_extraction(Lx, Ly, n_frames, filename: any, **suite2p_params) -> dict:
    """
    Perform Suite2p extraction on the provided imaging data.

    Parameters
    ----------    
    Lx : int
        The width of the imaging data.
    Ly : int
        The height of the imaging data.
    n_frames : int
        The number of frames in the imaging data.     
    filename : any
        path to the raw imaging data file.
    **suite2p_params : dict
        Additional parameters to configure Suite2p processing.

    Returns
    -------
    dict
        A dictionary containing the results of the Suite2p extraction.
    """
    # Initialize Suite2p with the provided parameters
    ops = default_ops()
    if suite2p_params:
        ops.update(suite2p_params)
    ops["save_path"] = "C:/Users/ariellel/data/suite2p_output/"
    with io.BinaryFile(Lx=Lx, Ly=Ly, filename=filename, n_frames=n_frames) as f_reg:
        pipeline(f_reg=f_reg, run_registration=False, ops=ops)
