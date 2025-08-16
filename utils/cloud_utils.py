import numpy as np
from scipy.ndimage import binary_fill_holes, label, find_objects, convolve
import os
import warnings
from typing import Union
from scipy.special import gamma, kv
from scipy.fftpack import fftn, ifftn

def generate_site_percolation_lattice(width: int, height: int, fill_prob: float, seed: int = None) -> np.ndarray:
    """
    Generate a binary site percolation lattice.

    Each site is independently filled (True) with probability `fill_prob`.
    The result is a 2D boolean NumPy array where True indicates a filled site.

    This function explicitly ensures the output uses the minimal memory footprint 
    for uncompressed boolean data (1 byte per site).

    Parameters:
    ----------
    width : int
        Number of columns in the lattice.
    height : int
        Number of rows in the lattice.
    fill_prob : float
        Probability that a given site is filled (between 0 and 1).
    seed : int 
        Seed used for randomized lattice generation. Default is None, in which case no seed is specified.

    Returns:
    -------
    np.ndarray
        A (height x width) boolean array (dtype=bool) representing the lattice.
    """
    rng = np.random.default_rng(seed)
    return (rng.random((height, width)) < fill_prob).astype(np.bool_)

def flood_fill_and_label_features(lattice: np.ndarray, connectivity: int = 1) -> tuple[np.ndarray, int]:
    """
    Flood fill enclosed regions in a boolean lattice and label connected features.

    This function:
    1. Fills all holes that are completely enclosed by True values.
    2. Labels each connected True region with a unique positive integer.

    Parameters:
    ----------
    lattice : np.ndarray
        A 2D boolean array representing the lattice.
    connectivity : int, optional (default=1)
        Connectivity for defining neighborhood:
        - 1: 4-connected (orthogonal neighbors)
        - 2: 8-connected (includes diagonals)

    Returns:
    -------
    tuple[np.ndarray, int]
        - A 2D integer array where each unique positive integer corresponds
          to a distinct connected feature (0 indicates background).
        - An integer count of the number of labeled features (clouds).
    """
    if lattice.dtype != np.bool_:
        raise ValueError("Input lattice must be of dtype bool.")

    # Step 1: Fill enclosed regions
    filled_lattice = binary_fill_holes(lattice)

    # Step 2: Label connected components
    structure = np.ones((3, 3)) if connectivity == 2 else None
    labeled_lattice, num_features = label(filled_lattice, structure=structure)

    return labeled_lattice, num_features

def save_lattice_npy(lattice: np.ndarray, directory: str, filename: str) -> None:
    """
    Save a NumPy lattice (2D array) as a `.npy` file in a specified directory.

    This is the most efficient uncompressed format for saving labeled or boolean lattices.

    Parameters:
    ----------
    lattice : np.ndarray
        The 2D NumPy array to save. Can be boolean or integer labeled lattice.
    directory : str
        The directory where the file should be saved. Will be created if it doesn't exist.
    filename : str
        The base filename (with or without `.npy` extension).
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not filename.endswith('.npy'):
        filename += '.npy'

    path = os.path.join(directory, filename)
    np.save(path, lattice)

def load_lattice_npy(directory: str, filename: str) -> np.ndarray:
    """
    Load a NumPy lattice (2D array) from a `.npy` file in the specified directory.

    Parameters:
    ----------
    directory : str
        Directory where the `.npy` file is stored.
    filename : str
        Filename of the saved lattice (with or without `.npy` extension).

    Returns:
    -------
    np.ndarray
        The 2D NumPy array (boolean or integer) that was saved.
    """
    if not filename.endswith('.npy'):
        filename += '.npy'

    path = os.path.join(directory, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Lattice file not found: {path}")

    return np.load(path)

def filter_labeled_features_by_size(
    labeled_lattice: np.ndarray,
    min_area: int = None,
    max_area: int = None
) -> tuple[np.ndarray, int]:
    """
    Filters a labeled lattice to keep only features within a specified area range.

    All features (non-zero labels) with area < `min_area` or area > `max_area`
    are removed (set to 0). Features within the area bounds are preserved.

    Parameters:
    ----------
    labeled_lattice : np.ndarray
        A 2D array where each unique positive integer represents a distinct feature.
    min_area : int, optional
        Minimum area (inclusive) required to keep a feature. If None, no lower bound.
    max_area : int, optional
        Maximum area (inclusive) allowed to keep a feature. If None, no upper bound.

    Returns:
    -------
    tuple[np.ndarray, int]
        - A filtered version of the labeled lattice with disqualified features removed.
        - The number of retained features after filtering.
    """
    if labeled_lattice.dtype.kind not in {'i', 'u'}:
        raise ValueError("Labeled lattice must be of integer type.")

    if min_area is None and max_area is None:
        # No filtering needed
        return labeled_lattice.copy(), np.max(labeled_lattice)

    if (min_area is not None and min_area < 0) or (max_area is not None and max_area < 0):
        raise ValueError("Area thresholds must be non-negative.")

    if min_area is not None and max_area is not None and max_area < min_area:
        raise ValueError("max_area must be greater than or equal to min_area.")

    # Count pixels for each label
    label_ids, counts = np.unique(labeled_lattice, return_counts=True)
    label_areas = dict(zip(label_ids, counts))
    label_areas.pop(0, None)  # Remove background label (0)

    # Determine which labels to keep
    labels_to_keep = [
        label for label, area in label_areas.items()
        if ((min_area is None or area >= min_area) and
            (max_area is None or area <= max_area))
    ]

    if not labels_to_keep:
        warnings.warn("No features matched the area filter criteria.")

    # Create filtered lattice
    mask = np.isin(labeled_lattice, labels_to_keep)
    filtered_lattice = np.where(mask, labeled_lattice, 0)

    return filtered_lattice, len(labels_to_keep)

def save_lattice_list_as_npy_files(
    lattice_list: list[np.ndarray],
    directory: str,
    prefix: str = "lattice"
) -> None:
    """
    Save a list of 2D NumPy lattices to `.npy` files in the specified directory.

    Each lattice is saved as an individual `.npy` file, using a filename based on
    the given prefix and its index in the list (e.g., lattice_000.npy, lattice_001.npy, ...).

    Parameters:
    ----------
    lattice_list : list[np.ndarray]
        A list of 2D NumPy arrays (boolean or integer type).
    directory : str
        Directory where the `.npy` files will be saved.
    prefix : str, optional
        Filename prefix for each saved lattice (default is "lattice").
    """
    num_digits = len(str(len(lattice_list) - 1)) if lattice_list else 1

    for i, lattice in enumerate(lattice_list):
        filename = f"{prefix}_{i:0{num_digits}d}"
        save_lattice_npy(lattice, directory, filename)

def compute_perimeter(cloud: np.ndarray) -> int:
    """
    Compute the perimeter of a binary cloud using 4-connected convolution.

    The perimeter is calculated as the number of exposed edges between True pixels
    and either:
    - False (background) pixels
    - the boundary of the array

    This is done by convolving a 4-connected kernel and counting for each True pixel
    how many of its 4 neighbors are empty or out-of-bounds.

    Parameters:
    ----------
    cloud : np.ndarray
        A 2D boolean array containing exactly one feature.

    Returns:
    -------
    int
        Total perimeter in pixel edge units.
    """
    if cloud.dtype != np.bool_:
        raise ValueError("Expected input to be a boolean array.")

    mask = cloud.astype(np.uint8)

    # 4-connected kernel: checks up, down, left, right
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.uint8)

    neighbor_counts = convolve(mask, kernel, mode='constant', cval=0)

    # Each True pixel contributes (4 - num_filled_neighbors) to the perimeter
    perimeter = np.sum(mask * (4 - neighbor_counts))

    return int(perimeter)

def compute_area(cloud: np.ndarray) -> int:
    """
    Compute the area of a cropped cloud lattice.

    Area is defined as the number of True pixels (i.e., filled sites).

    Parameters:
    ----------
    cloud : np.ndarray
        A 2D boolean array containing exactly one feature.

    Returns:
    -------
    int
        Total area (number of True pixels).
    """

    return np.count_nonzero(cloud)

def _count_edge_contacts(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Count how many pixels in a given feature lie on each of the four edges 
    of the original lattice.

    Parameters:
    ----------
    coords : np.ndarray
        An (N, 2) array of (row, col) coordinates of the feature pixels, 
        in global (original lattice) coordinates.
    shape : tuple[int, int]
        The (height, width) of the original lattice.

    Returns:
    -------
    np.ndarray
        A length-4 array of counts: [left, top, right, bottom].
        Each entry counts how many feature pixels touch that edge.
    """
    rows, cols = coords[:, 0], coords[:, 1]
    height, width = shape
    return np.array([
        np.sum(cols == 0),             # left edge
        np.sum(rows == 0),             # top edge
        np.sum(cols == width - 1),     # right edge
        np.sum(rows == height - 1)     # bottom edge
    ])

def _classify_edge_contact(edge_counts: np.ndarray) -> str:
    """
    Interpret a cloud's edge contact pattern and assign a category string.

    Categories:
    - "internal"       : touches no edge
    - "single_edge"    : touches exactly one edge
    - "two_edge"       : touches two non-opposite edges
    - "non_mirrorable" : touches three or more edges, or two opposite edges

    Parameters:
    ----------
    edge_counts : np.ndarray
        A 4-element boolean array indicating which edges are touched.

    Returns:
    -------
    str
        One of the classification strings described above.
    """
    left, top, right, bottom = edge_counts > 0
    touched = [left, top, right, bottom]
    num_touched = sum(touched)

    if num_touched == 0:
        return "internal"
    if num_touched == 1:
        return "single_edge"
    if num_touched == 2:
        # Check for non-mirrorable case: opposite edges (top-bottom or left-right)
        if (left and right) or (top and bottom):
            return "non_mirrorable"
        return "two_edge"
    return "non_mirrorable"

def _is_contact_match(classification: str, contact_type: str) -> bool:
    """
    Determine whether a feature's edge contact classification matches the requested filter.

    Parameters:
    ----------
    classification : str
        One of the internal classifications: "internal", "single_edge", "two_edge", "non_mirrorable".
    contact_type : str
        The user's requested filter type. Options:
        - "internal"
        - "single_edge"
        - "two_edge"
        - "mirrorable"      (accepts "single_edge" or "two_edge")
        - "valid"           (accepts "internal" or "mirrorable")
        - "non_mirrorable"
        - "all"             (accepts everything)

    Returns:
    -------
    bool
        True if the feature should be included, False otherwise.
    """
    if contact_type == "all":
        return True
    if contact_type == "mirrorable":
        return classification in {"single_edge", "two_edge"}
    if contact_type == "valid":
        return classification in {"internal", "single_edge", "two_edge"}
    if contact_type == "non_mirrorable":
        return classification == "non_mirrorable"
    return classification == contact_type

def extract_cropped_clouds_by_size(
    labeled_lattice: np.ndarray,
    min_area: int = None,
    max_area: int = None,
    contact_type: str = "internal"
) -> list[np.ndarray]:
    """
    Extract tightly-cropped cloud lattices that match size and edge-contact constraints.

    Parameters:
    ----------
    labeled_lattice : np.ndarray
        A 2D labeled array (output of `scipy.ndimage.label`).
    min_area : int, optional
        Minimum pixel area required to retain a feature (inclusive).
    max_area : int, optional
        Maximum pixel area allowed for a feature (inclusive).
    contact_type : str, optional
        Which edge-contact types to allow. Options:
            - "internal" (touches no edge)
            - "single_edge" (touches exactly one edge)
            - "two_edge" (touches two edges, not opposite)
            - "mirrorable" (single_edge or two_edge)
            - "valid" (internal or mirrorable)
            - "non_mirrorable" (touches >= 3 edges or opposite edges) OR (all - valid)
            - "all" (no contact constraint; default)

    Returns:
    -------
    list[np.ndarray]
        A list of cropped 2D boolean arrays corresponding to filtered features.
    """
    if contact_type not in {
        "internal", "single_edge", "two_edge",
        "mirrorable", "valid", "non_mirrorable", "all"
    }:
        raise ValueError(f"Invalid contact_type '{contact_type}'.")

    bounds_check = lambda area: (
        (min_area is None or area >= min_area) and
        (max_area is None or area <= max_area)
    )

    cropped_clouds = []
    slices = find_objects(labeled_lattice)
    H, W = labeled_lattice.shape

    for label_id, obj_slice in enumerate(slices, start=1):
        if obj_slice is None:
            continue

        sub_lattice = labeled_lattice[obj_slice]
        mask = (sub_lattice == label_id)
        area = np.count_nonzero(mask)

        if not bounds_check(area):
            continue

        # Get original lattice coordinates of this cropped region
        rows, cols = np.where(mask)
        global_coords = np.stack([
            rows + obj_slice[0].start,
            cols + obj_slice[1].start
        ], axis=1)

        edge_counts = _count_edge_contacts(global_coords, (H, W))
        classification = _classify_edge_contact(edge_counts)

        if _is_contact_match(classification, contact_type):
            cropped_clouds.append(mask.astype(np.bool_))

    if not cropped_clouds:
        warnings.warn("No features matched the area and contact filter criteria.")

    return cropped_clouds

def slice_cloud_into_segments(
    cloud: np.ndarray,
    num_segments: int,
    min_col_width: int = 3
) -> Union[list[dict], bool]:
    """
    Slice a cloud lattice into vertical segments and compute metadata for each.

    Each segment has at least `min_col_width` columns. If not possible, the function
    returns False and emits a warning.

    For each segment, the following metadata is returned:
    - segment (2D bool array)
    - segment_id (int)
    - start_col (int)
    - end_col (int)
    - right_edge_mask (1D bool array)
    - naive_r_exposed (int)
    - shared_with_prev (int, only for i >= 1)

    Parameters:
    ----------
    cloud : np.ndarray
        A 2D boolean array representing a single cloud feature.
    num_segments : int
        Number of vertical segments to divide the cloud into.
    min_col_width : int, optional
        Minimum width (in columns) for each segment (default is 3).

    Returns:
    -------
    list of dict or bool
        List of segment metadata dictionaries, or False if slicing is infeasible.
    """
    if cloud.dtype != np.bool_:
        raise ValueError("Expected input cloud to be a boolean array.")

    h, w = cloud.shape
    if num_segments < 1:
        raise ValueError("Number of segments must be >= 1.")

    if w < num_segments * min_col_width:
        warnings.warn(
            f"Cloud width {w} too small to generate {num_segments} segments "
            f"with minimum width {min_col_width}."
        )
        return False

    base_width = w // num_segments
    remainder = w % num_segments

    segments = []
    left_edges = []
    right_edges = []

    start_col = 0
    for i in range(num_segments):
        slice_width = base_width + (1 if i < remainder else 0)
        end_col = start_col + slice_width

        segment = cloud[:, start_col:end_col]
        left_edge_mask = segment[:, 0]
        right_edge_mask = segment[:, -1]
        naive_r_exposed = np.count_nonzero(right_edge_mask)

        segment_data = {
            'segment_id': i,
            'segment': segment,
            'start_col': start_col,
            'end_col': end_col,
            'right_edge_mask': right_edge_mask,
            'naive_r_exposed': naive_r_exposed,
            # 'shared_with_prev' to be filled later (if i > 0)
        }

        segments.append(segment_data)
        left_edges.append(left_edge_mask)
        right_edges.append(right_edge_mask)

        start_col = end_col

    # Compute shared edge count between adjacent segments
    for i in range(1, num_segments):
        shared = np.count_nonzero(np.logical_and(right_edges[i - 1], left_edges[i]))
        segments[i]['shared_with_prev'] = shared

    return segments

def compute_mirrored_slice_geometry(segments: list[dict]) -> dict:
    """
    Given a list of cloud segments (from slice_cloud_into_segments), compute:
    - mirrored area and perimeter for each incremental slice (0 to N-1)
    - naive right exposed edge (boundary due to slicing)
    - raw (unmirrored) perimeter of each slice

    Returns metadata for each slice and the full cloud.

    Parameters:
    ----------
    segments : list of dict
        Output from slice_cloud_into_segments. Each dict must include:
            - 'segment' (np.ndarray)
            - 'segment_id' (int)
            - 'naive_r_exposed' (int)
            - 'shared_with_prev' (int), for i > 0

    Returns:
    -------
    dict
        {
            'full_area': int,
            'full_perimeter': int,
            'mirrored_slices': list of {
                'slice_id': int,
                'mirrored_area': int,
                'mirrored_perimeter': int,
                'exposed_edge_length': int,
                'slice_perimeter': int
            }
        }
    """
    mirrored_slices = []

    running_area = 0
    running_perim = 0
    running_shared = 0

    num_segments = len(segments)

    for i in range(num_segments):
        seg = segments[i]
        area_i = compute_area(seg['segment'])
        perim_i = compute_perimeter(seg['segment'])

        running_area += area_i
        running_perim += perim_i

        if i > 0:
            shared = segments[i].get('shared_with_prev', 0)
            running_shared += shared

        slice_perimeter = running_perim - 2 * running_shared
        naive_exposed = seg['naive_r_exposed']

        mirrored_area = 2 * running_area
        mirrored_perimeter = 2 * (slice_perimeter - naive_exposed)

        mirrored_slices.append({
            'slice_id': seg['segment_id'],
            'mirrored_area': mirrored_area,
            'mirrored_perimeter': mirrored_perimeter,
            'exposed_edge_length': naive_exposed,
            'slice_perimeter': slice_perimeter
        })

    return {
        'full_area': running_area,
        'full_perimeter': slice_perimeter,
        'mirrored_slices': mirrored_slices
    }

def float_to_filename_str(x: float) -> str:
    """
    Converts a float to a string suitable for filenames by replacing '.' with 'p'.

    Example:
        0.25 -> '0p25'
        1.0  -> '1p0'
    """
    return str(x).replace('.', 'p')

def flatten_cloud_metadata_for_csv(cloud_data_list):
    """
    Flattens cloud metadata into a list of dicts suitable for compact CSV storage.

    CSV format overview:
    - Each row corresponds to either:
        • The full cloud geometry (slice_id = -1), or
        • A single mirrored slice from that cloud.
    - To minimize redundant data, full cloud values are stored in the same columns
      as the per-slice data (e.g., 'mirrored_area', 'mirrored_perimeter'), even
      though they technically refer to the full cloud.
    - This allows us to reuse column names across all rows and keep the structure flat.

    Conventions:
    - slice_id = -1 indicates a row describing the full cloud geometry.
        * 'mirrored_area' contains the full cloud's area
        * 'mirrored_perimeter' contains the full cloud's perimeter
        * 'exposed_edge_length' and 'slice_perimeter' are set to -1 as placeholders
    - All other slice_id values (0, 1, 2, ...) represent individual mirrored slices.

    Output columns:
    - cloud_id: Unique ID for each cloud (based on list index)
    - slice_id: -1 for full cloud, otherwise index of the mirrored slice
    - mirrored_area: Area of the mirrored slice (or full cloud if slice_id = -1)
    - mirrored_perimeter: Perimeter of the mirrored slice (or full cloud if slice_id = -1)
    - exposed_edge_length: Length of the exposed edge in the original slice (or -1 for full cloud)
    - slice_perimeter: Pre-mirroring perimeter of the slice (or -1 for full cloud)
    """
    flattened = []

    for cloud_id, cloud in enumerate(cloud_data_list):
        # Row for full cloud (slice_id = -1, fields reused for storage efficiency)
        flattened.append({
            "cloud_id": cloud_id,
            "slice_id": -1,
            "mirrored_area": cloud["full_area"],
            "mirrored_perimeter": cloud["full_perimeter"],
            "exposed_edge_length": -1,
            "slice_perimeter": -1,
        })

        # Rows for each mirrored slice
        for slice_data in cloud["mirrored_slices"]:
            flattened.append({
                "cloud_id": cloud_id,
                "slice_id": slice_data["slice_id"],
                "mirrored_area": slice_data["mirrored_area"],
                "mirrored_perimeter": slice_data["mirrored_perimeter"],
                "exposed_edge_length": slice_data["exposed_edge_length"],
                "slice_perimeter": slice_data["slice_perimeter"],
            })

    return flattened

def generate_correlated_percolation_lattice(
    width: int,
    height: int,
    gamma_exp: float,
    p_val: float,
    seed: int = None
) -> np.ndarray:
    """
    Generate a binary correlated percolation lattice.

    Each site is filled (True) if the correlated field value is below p_val.
    The result is a 2D boolean NumPy array where True indicates a filled site.

    Parameters:
    ----------
    width : int
        Number of columns in the lattice.
    height : int
        Number of rows in the lattice.
    gamma_exp : float
        Correlation exponent for the field.
    p_val : float
        Threshold for filling sites.
    seed : int, optional
        Random seed.

    Returns:
    -------
    np.ndarray
        A (height x width) boolean array (dtype=bool) representing the lattice.
    """
    rng = np.random.default_rng(seed)
    L = max(width, height)
    kx = np.fft.fftfreq(width).reshape(-1, 1)
    ky = np.fft.fftfreq(height).reshape(1, -1)
    q2 = kx**2 + ky**2
    q = np.sqrt(q2)
    q[0, 0] = 1e-10  # avoid division by zero
    S_q = _compute_2d_spectral_density(q, gamma_exp)
    noise = rng.normal(0, 1, (height, width)) + 1j * rng.normal(0, 1, (height, width))
    hq = fftn(noise) * np.sqrt(S_q)
    field = np.real(ifftn(hq))
    field -= np.min(field)
    field /= np.max(field)
    return (field < p_val).astype(np.bool_)

def _compute_2d_spectral_density(q: np.ndarray, gamma_exp: float) -> np.ndarray:
    """
    Compute the 2D spectral density S(q) for a given wavenumber array and correlation exponent.

    This function is used in the generation of correlated percolation fields.
    It computes the spectral density S(q) as a function of the wavenumber magnitude q
    and the correlation exponent gamma_exp, using a formula involving the modified Bessel function.

    Optimized: Preserves original comments, no changes needed here.

    Parameters
    ----------
    q : np.ndarray
        2D array of wavenumber magnitudes (typically from np.fft.fftfreq).
    gamma_exp : float
        Correlation exponent (gamma) controlling the spatial correlation.

    Returns
    -------
    np.ndarray
        2D array of real spectral density values S(q), same shape as q.
        NaNs and infinities are replaced with zeros.
    """
    beta = (gamma_exp - 2) / 2
    q = np.where(q == 0, 1e-10, q)
    prefactor = (2 * np.pi) / gamma(beta + 1)
    S_q = prefactor * (q / 2) ** beta * kv(beta, q)
    return np.nan_to_num(np.real(S_q), nan=0.0, posinf=0.0, neginf=0.0)


import numpy as np
from scipy.fft import rfft2, irfft2
from scipy.special import kv, gamma
import gc

def _compute_2d_spectral_density_packed(qx: np.ndarray, qy: np.ndarray, gamma_exp: float) -> np.ndarray:
    """
    Compute the 2D spectral density S(q) in packed form for rfft2.

    Parameters
    ----------
    qx : np.ndarray
        1D array of frequencies along x (width direction).
    qy : np.ndarray
        1D array of frequencies along y (height direction).
    gamma_exp : float
        Correlation exponent (gamma) controlling the spatial correlation.

    Returns
    -------
    np.ndarray
        2D array of real spectral density values (height x (width//2 + 1)).
    """
    kx = qx.reshape(-1, 1)  # column vector
    ky = qy.reshape(1, -1)  # row vector
    q2 = kx**2 + ky**2
    q = np.sqrt(q2)
    q[0, 0] = 1e-10  # avoid division by zero

    beta = (gamma_exp - 2) / 2
    prefactor = (2 * np.pi) / gamma(beta + 1)
    S_q = prefactor * (q / 2) ** beta * kv(beta, q)

    return np.nan_to_num(np.real(S_q), nan=0.0, posinf=0.0, neginf=0.0)

def generate_correlated_percolation_lattice_optimized(
    width: int,
    height: int,
    gamma_exp: float,
    p_val: float,
    seed: int = None
) -> np.ndarray:
    """
    Generate a binary correlated percolation lattice (optimized rfft2 version).

    This version uses only real noise, rfft2 packed representation to exploit Hermitian symmetry,
    and explicit memory cleanup for reduced peak usage.

    Parameters
    ----------
    width : int
        Number of columns in the lattice.
    height : int
        Number of rows in the lattice.
    gamma_exp : float
        Correlation exponent for the field.
    p_val : float
        Threshold for filling sites.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        A (height x width) boolean array (dtype=bool) representing the lattice.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate real noise
    noise = np.random.normal(0, 1, (height, width))

    # Compute rfft2 of noise (Hermitian symmetric packed)
    hq = rfft2(noise)

    # Delete noise to free memory
    del noise
    gc.collect()

    # Create packed spectral density shape
    qx = np.fft.fftfreq(height)
    qy = np.fft.rfftfreq(width)

    S_q = _compute_2d_spectral_density_packed(qx, qy, gamma_exp)

    # Apply spectral filter in packed form
    hq *= np.sqrt(S_q)

    # Delete spectral density to free memory
    del S_q
    gc.collect()

    # Inverse rfft2 to get real field
    field = irfft2(hq, s=(height, width))

    # Delete hq
    del hq
    gc.collect()

    # In-place normalization
    field_min = np.min(field)
    field -= field_min
    field_max = np.max(field)
    if field_max != 0:
        field /= field_max

    # Threshold to create lattice
    lattice = (field < p_val).astype(np.bool_)

    # Delete field
    del field
    gc.collect()

    return lattice
