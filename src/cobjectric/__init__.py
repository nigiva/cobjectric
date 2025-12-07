from importlib.metadata import version

__version__ = version("cobjectric")


def status() -> bool:
    """
    Check if the library is working
    
    Returns:
        bool: True if the library is working, False otherwise
    """
    return True