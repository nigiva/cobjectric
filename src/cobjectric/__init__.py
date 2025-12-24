from importlib.metadata import version

from cobjectric.base_model import BaseModel
from cobjectric.exceptions import CobjectricError
from cobjectric.sentinel import MissingValue

__version__ = version("cobjectric")

__all__ = ["BaseModel", "MissingValue", "CobjectricError"]


def status() -> bool:
    """
    Check if the library is working

    Returns:
        bool: True if the library is working, False otherwise
    """
    return True
