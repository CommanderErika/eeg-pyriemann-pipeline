import logging
from typing import Any, List, Optional, Type, Dict, TypeVar

from moabb import datasets, paradigms
from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm

# Set up logging to catch invalid names
logger = logging.getLogger(__name__)

# Define TypeVars for Generic function usage
T = TypeVar("T")

# Configuration / Registry
MOABB_DATASETS: Dict[str, Type[BaseDataset]] = {
    "Cho2017":      datasets.Cho2017,
    "Zhou2016":     datasets.Zhou2016,
    "BNCI2014_004": datasets.BNCI2014_004,
    "BNCI2014_002": datasets.BNCI2014_002,
    "BNCI2014_001": datasets.BNCI2014_001,
    "AlexMI":       datasets.AlexMI,
    "Liu2024":      datasets.Liu2024,
    "PhysionetMI":  datasets.PhysionetMI,
    "Lee2019_MI":   datasets.Lee2019_MI,
}

MOABB_PARADIGMS: Dict[str, Type[BaseParadigm]] = {
    "LeftRightImagery": paradigms.LeftRightImagery,
    "MotorImagery":     paradigms.MotorImagery,
    "FilterBankLeftRightImagery": paradigms.FilterBankLeftRightImagery
}

def get_component(name: str, registry: Dict[str, Type[T]]) -> Optional[Type[T]]:
    """
    Generic getter for Datasets or Paradigms.
    
    Args:
        name: The key name to look up.
        registry: The dictionary containing the mapping.
        
    Returns:
        The class type if found, else None.
    """
    if name not in registry:
        logger.warning(f"Component '{name}' not found in registry.")
        return None
    return registry[name]

def handle_components(names: List[str], registry: Dict[str, Type[T]]) -> List[Type[T]]:
    """
    Generic handler to validate and retrieve a list of components.
    
    Args:
        names: List of string names to retrieve.
        registry: The dictionary to look them up in.
        
    Returns:
        List of valid Class Types.
    """
    valid_components = []
    
    for name in names:
        component = get_component(name, registry)
        if component:
            valid_components.append(component)
            
    return valid_components

# --- Facade Functions (API Wrappers) ---

def get_dataset(dataset_name: str) -> Optional[Type[BaseDataset]]:
    return get_component(dataset_name, MOABB_DATASETS)

def get_paradigm(paradigm_name: str) -> Optional[Type[BaseParadigm]]:
    return get_component(paradigm_name, MOABB_PARADIGMS)

def handle_datasets(dataset_names: List[str]) -> List[Type[BaseDataset]]:
    """Filters valid datasets and logs warnings for invalid ones."""
    return handle_components(dataset_names, MOABB_DATASETS)

def handle_paradigms(paradigm_names: List[str]) -> List[Type[BaseParadigm]]:
    """Filters valid paradigms and logs warnings for invalid ones."""
    return handle_components(paradigm_names, MOABB_PARADIGMS)