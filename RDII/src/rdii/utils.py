import json
import os


def load_config(config_path ):
    """
    Load pipeline configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return json.load(f)
    
def detect_n_workers(n_workers_requested):
    """
    Detect number of workers based on environment.

    Parameters
    ----------
    n_workers_requested : int
        Maximum number of workers requested.

    Returns
    -------
    int
        Number of workers to use.
    """
    if os.environ.get("SLURM_NTASKS"):
        return min(n_workers_requested, int(os.environ["SLURM_NTASKS"]))
    return min(2, n_workers_requested)