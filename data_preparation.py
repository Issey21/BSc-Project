import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import h5py
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, Callable


# The obervation condition class for other function.
class ObservingConditions:
    def __init__(self, observing_conditions: Dict[str, np.ndarray]):
        # Ensure all values are numpy arrays and have consistent lengths
        if not all(isinstance(v, np.ndarray) for v in observing_conditions.values()):
            raise ValueError("All observing conditions must be numpy arrays.")
        
        first_length = len(next(iter(observing_conditions.values())))
        if not all(len(v) == first_length for v in observing_conditions.values()):
            raise ValueError("All observing conditions must have the same length.")

        self.observing_conditions = observing_conditions

    def as_array(self) -> np.ndarray:
        """Return observing conditions as a 2D numpy array."""
        return np.hstack([v.reshape(-1, 1) for v in self.observing_conditions.values()])

    def keys(self):
        """Return the keys of the observing conditions."""
        return self.observing_conditions.keys()

    def __getitem__(self, key):
        """Access individual observing condition arrays by key."""
        return self.observing_conditions[key]
    
def load_metadata(name_or_path: Union[str, Path], **_: Any) -> dict:
    """
    Load the metadata.

    Args:
        name_or_path: Name of a data set (e.g., `"beta_pictoris__lp"`),
            or Path to an HDF file that contains the data set.

    Returns:
        A dictionary containing the metadata.
    """

    # Initialize metadata
    metadata: Dict[str, Union[str, float]] = dict()

    # Get the path to the HDF file that contains the data to be loaded
    file_path = _resolve_name_or_path(name_or_path)

    # Read in the data set from the HDF file
    with h5py.File(file_path, 'r') as hdf_file:
        for key in hdf_file['metadata'].keys():
            value = hdf_file['metadata'][key][()]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            metadata[key] = value

    return metadata
    

# Helper function to resolve a path from a string or Path object
def _resolve_name_or_path(name_or_path: Union[str, Path]) -> Path:
    if isinstance(name_or_path, str):
        return Path(name_or_path)
    return name_or_path


# Function to stack arrays along a specified axis with an optional stacking function
def prestack_array(
    array: np.ndarray,
    stacking_factor: int,
    stacking_function: Callable = np.mean,
    axis: int = 0,
) -> np.ndarray:
    if stacking_factor == 1:
        return array
    n_splits = np.ceil(array.shape[axis] / stacking_factor).astype(int)
    split_indices = [i * stacking_factor for i in range(1, n_splits)]
    return np.stack(
        [
            stacking_function(block, axis=axis)
            for block in np.split(array, split_indices, axis=axis)
        ],
        axis=axis,
    )


# Function to load a stack of images from an HDF5 file
def load_stack(
    name_or_path: Union[str, Path],
    binning_factor: int = 1,
    frame_size: Optional[Tuple[int, int]] = None,
    remove_planets: bool = False,
) -> np.ndarray:
    file_path = _resolve_name_or_path(name_or_path)
    with h5py.File(file_path, 'r') as hdf_file:
        stack_shape = hdf_file['stack'].shape
        if frame_size is not None:
            target_shape = (-1, frame_size[0], frame_size[1])
        else:
            target_shape = (-1, -1, -1)
        slices = [slice(None)] * len(stack_shape)
        for i, (old_len, new_len) in enumerate(zip(stack_shape, target_shape)):
            if new_len > old_len:
                start = None
                end = None
            else:
                start = old_len // 2 - new_len // 2 if new_len != -1 else None
                end = start + new_len if start is not None else None
            slices[i] = slice(start, end)
        stack = np.array(hdf_file['stack'][tuple(slices)], dtype=np.float32)
    stack = prestack_array(array=stack, stacking_factor=binning_factor)
    return stack


# Function to load parallactic angles from an HDF5 file
def load_parang(name_or_path: Union[str, Path], binning_factor: int = 1) -> np.ndarray:
    file_path = _resolve_name_or_path(name_or_path)
    with h5py.File(file_path, 'r') as hdf_file:
        parang = np.array(hdf_file['parang'][:], dtype=np.float32)
    return prestack_array(parang, binning_factor)


# Function to load the dataset including stack and parallactic angles
def load_dataset(
    name_or_path: Union[str, Path],
    binning_factor: int = 1,
    frame_size: Optional[Tuple[int, int]] = None,
    remove_planets: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    stack = load_stack(
        name_or_path=name_or_path,
        binning_factor=binning_factor,
        frame_size=frame_size,
        remove_planets=remove_planets,
    )
    parang = load_parang(
        name_or_path=name_or_path,
        binning_factor=binning_factor,
    )
    psf_template = load_psf_template(
        name_or_path=name_or_path,
    )
    observing_conditions = load_observing_conditions(
        name_or_path=name_or_path,
        binning_factor=binning_factor,
    )
    metadata = load_metadata(
        name_or_path=name_or_path,
    )

    return stack, parang, psf_template, observing_conditions, metadata


def load_psf_template(name_or_path: Union[str, Path], **_: Any) -> np.ndarray:
    """
    Load the unsaturated PSF template.

    Args:
        name_or_path: Name of a data set (e.g., `"beta_pictoris__lp"`),
            or Path to an HDF file that contains the data set.

    Returns:
        A numpy array containing the unsaturated PSF template.
    """

    # Get the path to the HDF file that contains the data to be loaded
    file_path = _resolve_name_or_path(name_or_path)

    # Read in the data set from the HDF file
    with h5py.File(file_path, 'r') as hdf_file:
        psf_template = np.array(hdf_file['psf_template'], dtype=np.float32)
        psf_template = psf_template.squeeze()

    # Ensure that the PSF template is two-dimensional now; otherwise this
    # can result in weird errors that are hard to debug
    if psf_template.ndim != 2:
        raise RuntimeError(
            f'psf_template is not 2D! (shape = {psf_template.shape})'
        )

    return psf_template


def load_observing_conditions(
    name_or_path: Union[str, Path],
    binning_factor: int = 1,
    **_: Any,
) -> ObservingConditions:
    """
    Load the observing conditions.

    Args:
        name_or_path: Name of a data set (e.g., `"beta_pictoris__lp"`),
            or Path to an HDF file that contains the data set.
        binning_factor: Number of time steps that should be temporally
            binned ("pre-stacked") using a block-wise mean.

    Returns:
        An object containing the observing conditions.
    """

    # Get the path to the HDF file that contains the data to be loaded
    file_path = _resolve_name_or_path(name_or_path)

    # Read in the data set from the HDF file
    with h5py.File(file_path, 'r') as hdf_file:

        # Collect the observing conditions into a (temporary) dictionary
        _observing_conditions = dict()
        for key in hdf_file['observing_conditions']['interpolated'].keys():
            _observing_conditions[key] = np.array(
                hdf_file['observing_conditions']['interpolated'][key],
                dtype=np.float32,
            )

    # Apply temporal binning to the observing conditions
    for key in _observing_conditions.keys():
        _observing_conditions[key] = prestack_array(
            array=_observing_conditions[key], stacking_factor=binning_factor
        )

    # Convert the observing conditions into an ObservingConditions object
    observing_conditions = ObservingConditions(_observing_conditions)

    return observing_conditions


def get_field_rotation(parang: np.ndarray) -> float:
    """
    Compute the field rotation from a given array of parallactic angles.

    Args:
        parang: A 1D numpy array of shape `(n_frames,)` that contains
            the parallactic angle for each frame in degree.

    Returns:
        The field rotation in degree.
    """

    # Compute field rotation
    field_rotation = float(abs(parang[-1] - parang[0]))

    # If the value is physically sensible, we can return it
    if field_rotation < 180:
        return field_rotation

    # If the observation crosses the zenith, we sometimes get "jumps" in the
    # parallactic angle that break the computation of the field rotation. By
    # ensuring all angles are positive, we can try to fix this issue.
    new_parang = (parang + 360) % 360
    field_rotation = abs(new_parang[-1] - new_parang[0])
    if field_rotation < 180:
        return field_rotation

    # If the value is still not physically sensible, raise an error
    raise RuntimeError('Field rotation is greater than 180 degrees!')



# Custom PyTorch Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert data to PyTorch tensor
        self.labels = torch.tensor(labels, dtype=torch.float32)  # Convert labels to PyTorch tensor

    def __len__(self):
        return len(self.data)  # Return the number of samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # Return a sample and its label


# Normalize the data
def normalize_data(data: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(data)



def prepare_data(file_path: Union[str, Path], batch_size: int = 32, shuffle: bool = True, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
    # Load dataset components
    stack, parang, psf_template, observing_conditions, metadata = load_dataset(name_or_path=file_path)

    # Reshape stack for input into the model (flatten images)
    stack_flattened = stack.reshape(stack.shape[0], -1)

    # Normalize the data
    stack_normalized = normalize_data(stack_flattened)
    
    # Example: Assigning labels based on parallactic angles
    labels = parang  # or another suitable label

    # Split data into train, validation, and test sets
    num_samples = len(stack_normalized)
    indices = list(range(num_samples))
    split_train = int(split_ratios[0] * num_samples)
    split_val = int(split_ratios[1] * num_samples)
    
    train_indices = indices[:split_train]
    val_indices = indices[split_train:split_train + split_val]
    test_indices = indices[split_train + split_val:]
    
    train_data, val_data, test_data = stack_normalized[train_indices], stack_normalized[val_indices], stack_normalized[test_indices]
    train_labels, val_labels, test_labels = labels[train_indices], labels[val_indices], labels[test_indices]

    # Create PyTorch datasets
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Return data loaders and input size (flattened image size)
    return train_loader, val_loader, test_loader, stack_normalized.shape[1]


# def prepare_data(file_path: str, binning_factor: int = 1, frame_size: Optional[Tuple[int, int]] = None):
    # Load the stack, parang, PSF template, and observing conditions
    stack, parang = load_dataset(
        name_or_path=file_path,
        binning_factor=binning_factor,
        frame_size=frame_size
    )
    psf_template = load_psf_template(name_or_path=file_path)
    observing_conditions = load_observing_conditions(name_or_path=file_path, binning_factor=binning_factor)

    # Apply the PSF correction to the stack
    stack_corrected = correct_images_with_psf(stack, psf_template)

    # Ensure stack_corrected is properly defined before reshaping
    if stack_corrected is None:
        raise ValueError("PSF correction did not return any data.")

    # Flatten the images: from (N, height, width) to (N, height * width)
    stack_flattened = stack_corrected.reshape(stack_corrected.shape[0], -1)


    # Include observing conditions as additional features
    obs_conditions_array = observing_conditions.as_array()  # Directly get the array representation
    stack_with_conditions = np.hstack((stack_flattened, obs_conditions_array))

    # Normalize the data
    scaler = StandardScaler()
    stack_normalized = scaler.fit_transform(stack_with_conditions)

    # Prepare labels
    labels = np.array(parang)

    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(stack_normalized, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create PyTorch datasets and dataloaders
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = stack_normalized.shape[1]  # Updated to include the size after adding observing conditions

    return train_loader, val_loader, test_loader, input_size