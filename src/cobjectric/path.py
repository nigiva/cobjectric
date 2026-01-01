def parse_path(path: str) -> list[str]:
    """
    Parse a path string into segments.

    Args:
        path: Path string (e.g., "address.city", "items[0].name").

    Returns:
        List of path segments (e.g., ["address", "city"] or ["items", "[0]", "name"]).

    Raises:
        KeyError: If the path is invalid.
    """
    segments: list[str] = []
    current = ""
    i = 0
    while i < len(path):
        if path[i] == ".":
            if current:
                segments.append(current)
                current = ""
        elif path[i] == "[":
            if current:
                segments.append(current)
                current = ""
            # Find closing bracket
            j = i + 1
            while j < len(path) and path[j] != "]":
                j += 1
            if j >= len(path):
                raise KeyError(f"Invalid path: {path}")
            index_str = path[i + 1 : j]
            try:
                index = int(index_str)
                segments.append(f"[{index}]")
            except ValueError as e:
                raise KeyError(f"Invalid path: {path}") from e
            i = j
        else:
            current += path[i]
        i += 1
    if current:
        segments.append(current)
    return segments
