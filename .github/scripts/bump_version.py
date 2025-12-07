#!/usr/bin/env python3
"""
Version bumping script for CI/CD.

This script reads the current version from pyproject.toml and calculates
the next version based on the release type.

Usage:
    python bump_version.py <release_type>

Release types:
    - major: 0.1.0 -> 1.0.0
    - minor: 0.1.0 -> 0.2.0
    - patch: 0.1.0 -> 0.1.1
    - premajor: 0.1.0 -> 1.0.0a0
    - preminor: 0.1.0 -> 0.2.0a0
    - prepatch: 0.1.0 -> 0.1.1a0
"""

import re
import sys
from pathlib import Path


def parse_version(version_str: str) -> tuple[int, int, int, str | None, int | None]:
    """
    Parse a version string into its components.

    Args:
        version_str (str): Version string (e.g., "1.2.3" or "1.2.3a0")

    Returns:
        tuple[int, int, int, str | None, int | None]: (major, minor, patch, prerelease_type, prerelease_num)
    """
    pattern = r"^(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?$"
    match = re.match(pattern, version_str)

    if not match:
        raise ValueError(f"Invalid version format: {version_str}")

    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3))
    prerelease_type = match.group(4)
    prerelease_num = int(match.group(5)) if match.group(5) else None

    return major, minor, patch, prerelease_type, prerelease_num


def format_version(
    major: int,
    minor: int,
    patch: int,
    prerelease_type: str | None = None,
    prerelease_num: int | None = None,
) -> str:
    """
    Format version components into a version string.

    Args:
        major (int): Major version number.
        minor (int): Minor version number.
        patch (int): Patch version number.
        prerelease_type (str | None): Prerelease type (a, b, rc) or None.
        prerelease_num (int | None): Prerelease number or None.

    Returns:
        str: Formatted version string.
    """
    version = f"{major}.{minor}.{patch}"
    if prerelease_type and prerelease_num is not None:
        version += f"{prerelease_type}{prerelease_num}"
    return version


def bump_version(current_version: str, release_type: str) -> str:
    """
    Calculate the next version based on the release type.

    Args:
        current_version (str): Current version string.
        release_type (str): Type of release (major, minor, patch, premajor, preminor, prepatch).

    Returns:
        str: New version string.
    """
    major, minor, patch, pre_type, pre_num = parse_version(current_version)

    # If current version is a prerelease and we're doing a stable release,
    # just remove the prerelease suffix
    is_prerelease = pre_type is not None

    if release_type == "major":
        if is_prerelease and minor == 0 and patch == 0:
            # 1.0.0a0 -> 1.0.0
            return format_version(major, 0, 0)
        return format_version(major + 1, 0, 0)

    elif release_type == "minor":
        if is_prerelease and patch == 0:
            # 0.2.0a0 -> 0.2.0
            return format_version(major, minor, 0)
        return format_version(major, minor + 1, 0)

    elif release_type == "patch":
        if is_prerelease:
            # 0.1.1a0 -> 0.1.1
            return format_version(major, minor, patch)
        return format_version(major, minor, patch + 1)

    elif release_type == "premajor":
        if is_prerelease and minor == 0 and patch == 0:
            # 1.0.0a0 -> 1.0.0a1
            return format_version(major, 0, 0, "a", pre_num + 1)
        return format_version(major + 1, 0, 0, "a", 0)

    elif release_type == "preminor":
        if is_prerelease and patch == 0:
            # 0.2.0a0 -> 0.2.0a1
            return format_version(major, minor, 0, "a", pre_num + 1)
        return format_version(major, minor + 1, 0, "a", 0)

    elif release_type == "prepatch":
        if is_prerelease:
            # 0.1.1a0 -> 0.1.1a1
            return format_version(major, minor, patch, "a", pre_num + 1)
        return format_version(major, minor, patch + 1, "a", 0)

    else:
        raise ValueError(f"Unknown release type: {release_type}")


def get_current_version() -> str:
    """
    Read the current version from pyproject.toml.

    Returns:
        str: Current version string.
    """
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"

    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    content = pyproject_path.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)

    if not match:
        raise ValueError("Could not find version in pyproject.toml")

    return match.group(1)


def main() -> None:
    """
    Main entry point for the version bump script.
    """
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <release_type>", file=sys.stderr)
        print(
            "Release types: major, minor, patch, premajor, preminor, prepatch",
            file=sys.stderr,
        )
        sys.exit(1)

    release_type = sys.argv[1].lower()
    valid_types = {"major", "minor", "patch", "premajor", "preminor", "prepatch"}

    if release_type not in valid_types:
        print(f"Error: Invalid release type '{release_type}'", file=sys.stderr)
        print(f"Valid types: {', '.join(sorted(valid_types))}", file=sys.stderr)
        sys.exit(1)

    try:
        current_version = get_current_version()
        new_version = bump_version(current_version, release_type)
        # Output only the new version (used by CI)
        print(new_version)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
