# Release Guide

This document explains how to create a new release for this project. The release process is automated through GitHub Actions workflows and follows Semantic Versioning (SemVer) principles.

## Overview

The release process is triggered automatically when a pull request with a branch name starting with `release/` is merged into the `main` branch. The workflow will:

1. Extract the release type from the branch name
2. Calculate the new version number
3. Update version files (`pyproject.toml`, `README.md`)
4. Create a Git tag
5. Build the package
6. Publish to PyPI
7. Create a GitHub Release
8. Sync the `dev` branch with `main`

## Release Types

The project uses Semantic Versioning with the format `MAJOR.MINOR.PATCH` (e.g., `1.2.3`). Additionally, pre-release versions are supported with the format `MAJOR.MINOR.PATCHaN` (e.g., `1.2.3a0`).

### Stable Releases

#### `major`

Increments the major version number and resets minor and patch to zero. Use this for breaking changes that are incompatible with previous versions.

**Examples:**
- `0.1.0` → `1.0.0`
- `1.2.3` → `2.0.0`
- `1.0.0a0` → `1.0.0` (removes prerelease suffix if already at major.0.0)

**When to use:**
- Breaking API changes
- Removal of deprecated features
- Major architectural changes
- Incompatible dependency updates

#### `minor`

Increments the minor version number and resets patch to zero. Use this for new features that are backward compatible.

**Examples:**
- `0.1.0` → `0.2.0`
- `1.2.3` → `1.3.0`
- `0.2.0a0` → `0.2.0` (removes prerelease suffix if already at X.minor.0)

**When to use:**
- New features added
- New functionality that doesn't break existing code
- Deprecation warnings (without removal)
- Significant improvements to existing features

#### `patch`

Increments the patch version number. Use this for bug fixes and small changes that are backward compatible.

**Examples:**
- `0.1.0` → `0.1.1`
- `1.2.3` → `1.2.4`
- `0.1.1a0` → `0.1.1` (removes prerelease suffix)

**When to use:**
- Bug fixes
- Security patches
- Documentation updates
- Minor performance improvements
- Small refactoring that doesn't change behavior

### Pre-releases

Pre-releases are versions that are not yet stable. They are useful for testing and early access. Pre-release versions use the format `MAJOR.MINOR.PATCHaN` where `N` is an incrementing number.

#### `premajor`

Creates a pre-release version for the next major version. If the current version is already a prerelease at `X.0.0aN`, it increments the prerelease number.

**Examples:**
- `0.1.0` → `1.0.0a0` (first premajor prerelease)
- `1.0.0a0` → `1.0.0a1` (increment prerelease number)
- `1.2.3` → `2.0.0a0` (new major version with prerelease)

**When to use:**
- Testing major breaking changes
- Early access to major version features
- Gathering feedback before a major release

#### `preminor`

Creates a pre-release version for the next minor version. If the current version is already a prerelease at `X.Y.0aN`, it increments the prerelease number.

**Examples:**
- `0.1.0` → `0.2.0a0` (first preminor prerelease)
- `0.2.0a0` → `0.2.0a1` (increment prerelease number)
- `1.2.3` → `1.3.0a0` (new minor version with prerelease)

**When to use:**
- Testing new features before stable release
- Early access to minor version features
- Beta testing of new functionality

#### `prepatch`

Creates a pre-release version for the next patch version. If the current version is already a prerelease, it increments the prerelease number.

**Examples:**
- `0.1.0` → `0.1.1a0` (first prepatch prerelease)
- `0.1.1a0` → `0.1.1a1` (increment prerelease number)
- `1.2.3` → `1.2.4a0` (new patch version with prerelease)

**When to use:**
- Testing bug fixes before stable release
- Early access to patch fixes
- Quick iteration on fixes

## How to Create a Release

### Step 1: Ensure Code is Ready

Before creating a release, make sure:

1. All changes are committed and pushed to the `dev` branch
2. All tests pass: `uv run inv test`
3. Code quality checks pass: `uv run inv all`
4. The code has been reviewed and approved

> **Important**: If tests or code quality checks fail, the CI workflow will fail and block the release. The pull request cannot be merged until all CI checks pass successfully.

### Step 2: Create a Release Branch

> **Note**: You do not need to manually create a Git tag or update version numbers. The GitHub Actions workflow will automatically handle:
> - Creating the Git tag
> - Updating the version in `pyproject.toml`, `README.md`, and `uv.lock`
> - Committing these changes to `main` (this creates an additional commit on `main` that won't be in `dev`)
> - The `dev` branch will be automatically rebased from `main` after the release to include this version bump commit

Create a new branch from `main` (or from `dev` if you want to include the latest changes) with the following naming convention:

```
release/{type}/{description}
```

Where:
- `{type}` is one of: `major`, `minor`, `patch`, `premajor`, `preminor`, `prepatch`
- `{description}` is a short description (optional but recommended)

**Examples:**
```bash
# Create a patch release
git checkout main
git pull origin main
git checkout -b release/patch/bug-fixes

# Create a minor release
git checkout -b release/minor/new-features

# Create a major release
git checkout -b release/major/api-changes

# Create a pre-release
git checkout -b release/preminor/beta-testing
```

### Step 3: Create a Pull Request

1. Push the release branch to GitHub:
   ```bash
   git push origin release/{type}/{description}
   ```

2. Create a pull request from the release branch to `main`

3. Fill in the pull request details:
   - **Title**: Should describe the release (e.g., "Release v1.2.3: Bug fixes and improvements")
   - **Description**: This description will be used as the GitHub Release description. It should include:
     - Summary of changes
     - List of new features (for minor/major releases)
     - List of bug fixes (for patch releases)
     - Breaking changes (for major releases)
     - Migration guide if applicable

4. Wait for CI checks to pass:
   - Code quality checks
   - Tests and coverage
   - All checks must pass before merging

### Step 4: Merge the Pull Request

Once the pull request is approved and all CI checks pass:

1. Merge the pull request into `main`
2. The release workflow will automatically trigger

### Step 5: Monitor the Release Workflow

The GitHub Actions workflow will automatically:

1. **Extract release type** from the branch name
2. **Calculate new version** using the current version in `pyproject.toml`
3. **Check if tag exists** - if the tag already exists, the workflow will skip version bumping
4. **Update version files**:
   - `pyproject.toml` - updates the `version` field
   - `README.md` - updates any version references (if present)
   - `uv.lock` - updates the lock file
5. **Commit version bump** to `main` branch
6. **Create and push Git tag** (e.g., `v1.2.3`)
7. **Build the package** using `uv build`
8. **Publish to PyPI** (requires authentication)
9. **Create GitHub Release** with:
   - Title: The version tag (e.g., `v1.2.3`)
   - Notes: Pull request title and body
   - Assets: Built package files
10. **Sync dev branch** - rebases `dev` from `main`

### Step 6: Verify the Release

After the workflow completes, verify:

1. **GitHub Release**: Check that the release was created at `https://github.com/{owner}/{repo}/releases`
2. **PyPI**: Verify the package is available at `https://pypi.org/project/cobjectric/`
3. **Git Tag**: Confirm the tag exists: `git fetch --tags && git tag -l`
4. **Version Files**: Check that `pyproject.toml` has the correct version

## Workflow Details

### Release Workflow Trigger

The release workflow (`.github/workflows/release.yml`) triggers when:
- A pull request is **closed** (merged)
- The PR targets the `main` branch
- The PR's source branch name starts with `release/`

### Version Calculation Logic

The version bumping script (`.github/scripts/bump_version.py`) implements the following logic:

#### From Stable Versions

- **major**: `X.Y.Z` → `(X+1).0.0`
- **minor**: `X.Y.Z` → `X.(Y+1).0`
- **patch**: `X.Y.Z` → `X.Y.(Z+1)`
- **premajor**: `X.Y.Z` → `(X+1).0.0a0`
- **preminor**: `X.Y.Z` → `X.(Y+1).0a0`
- **prepatch**: `X.Y.Z` → `X.Y.(Z+1)a0`

#### From Pre-release Versions

- **major** (from `X.0.0aN`): `X.0.0aN` → `X.0.0` (removes prerelease)
- **minor** (from `X.Y.0aN`): `X.Y.0aN` → `X.Y.0` (removes prerelease)
- **patch** (from `X.Y.ZaN`): `X.Y.ZaN` → `X.Y.Z` (removes prerelease)
- **premajor** (from `X.0.0aN`): `X.0.0aN` → `X.0.0a(N+1)` (increments prerelease)
- **preminor** (from `X.Y.0aN`): `X.Y.0aN` → `X.Y.0a(N+1)` (increments prerelease)
- **prepatch** (from `X.Y.ZaN`): `X.Y.ZaN` → `X.Y.Za(N+1)` (increments prerelease)

### Safety Checks

The workflow includes several safety mechanisms:

1. **Tag Existence Check**: If a tag for the calculated version already exists, the workflow skips version bumping and tag creation to prevent duplicate releases
2. **Release Existence Check**: Before creating a GitHub Release, it checks if one already exists
3. **PyPI Skip Existing**: The PyPI publish step uses `skip-existing: true` to avoid errors if the version is already published

## Troubleshooting

### Tag Already Exists

If the tag already exists, the workflow will skip the version bump. This can happen if:
- The release was partially completed before
- The tag was created manually

**Solution**: Either:
- Use a different release type (e.g., `patch` instead of `minor`)
- Manually delete the tag if it was created incorrectly
- Continue with the existing tag (the workflow will still build and publish)

### Workflow Fails After Version Bump

If the workflow fails after updating the version but before completing:
- The version in `pyproject.toml` will already be updated
- You may need to manually revert the version or create a new release branch

**Solution**: Check the workflow logs to identify the failure, fix the issue, and create a new release branch if needed.

### PyPI Authentication Issues

If publishing to PyPI fails:
- Ensure the `release` environment has the correct permissions
- Check that the PyPI token is configured correctly in GitHub Secrets

### Dev Branch Sync Fails

The `sync-dev` job has `continue-on-error: true`, so it won't fail the entire workflow. If it fails:
- Manually rebase `dev` from `main`:
  ```bash
  git checkout dev
  git pull origin dev
  git rebase origin/main
  git push origin dev --force-with-lease
  ```

## Best Practices

1. **Always test locally** before creating a release branch
2. **Use descriptive branch names** to make it clear what the release contains
3. **Write comprehensive PR descriptions** - they become the release notes
4. **Review the calculated version** in the workflow logs before merging
5. **Wait for CI to pass** before merging the release PR
6. **Monitor the workflow** after merging to ensure it completes successfully
7. **Test the published package** after release to verify it works correctly

## Examples

### Example 1: Patch Release for Bug Fixes

```bash
# Current version: 1.2.3
git checkout main
git pull origin main
git checkout -b release/patch/critical-bug-fixes

# Make sure all changes are committed
git push origin release/patch/critical-bug-fixes

# Create PR, merge when ready
# Result: Version becomes 1.2.4
```

### Example 2: Minor Release for New Features

```bash
# Current version: 1.2.3
git checkout main
git pull origin main
git checkout -b release/minor/user-authentication

# Create PR with description of new features
# Result: Version becomes 1.3.0
```

### Example 3: Pre-release for Beta Testing

```bash
# Current version: 1.2.3
git checkout main
git pull origin main
git checkout -b release/preminor/beta-features

# Create PR for beta testing
# Result: Version becomes 1.3.0a0

# Later, when ready for stable release:
git checkout -b release/minor/stable-release
# Result: Version becomes 1.3.0 (removes prerelease suffix)
```

## Related Documentation

- [Semantic Versioning](https://semver.org/) - Official SemVer specification
- [GitHub Actions Workflows](.github/workflows/) - Workflow definitions
- [Version Bump Script](.github/scripts/bump_version.py) - Version calculation logic

