# GitHub Actions Workflow for Python CI

## Overview

This repository uses GitHub Actions for continuous integration. The workflow runs automatically on push to the `main` branch and on pull requests to ensure code quality and correctness.

## CI Workflow (`ci.yml`)

The workflow performs the following checks:

1. **Environment Setup**
   - Sets up Python 3.13
   - Installs the UV package manager
   - Installs all project dependencies

2. **Code Quality Checks**
   - **Linting**: Runs `ruff check` to identify and report code quality issues
   - **Formatting**: Verifies code follows consistent formatting with `ruff format --check`
   - **Type Checking**: Ensures type correctness with `ty check`

3. **Testing**
   - Runs all tests with pytest

## Local Development

You can run the same checks locally before pushing using the following commands:

```bash
# Format code
uvx ruff format .

# Run linting
uvx ruff check .

# Type checking
uvx ty check .

# Run tests
uv run pytest tests
```

## Workflow Status

Check the Actions tab in the GitHub repository to view the status and logs of workflow runs.