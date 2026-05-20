# Local Environment Setup Guide (Standard `uv` Workflow)

This document outlines the professional workflow for managing dependencies using `uv`. This setup ensures **Qwen 3.5** support and consistent environment states across different machines.

## Prerequisites

- **Python**: 3.11 or higher
- **uv**: High-performance Python package manager (`pip install uv`)

## Installation & Initialization

### 1. Initialize the Project
If you haven't already, initialize the project to create a `pyproject.toml` file. This is the "Source of Truth" for your dependencies.

```bash
uv init
```

### 2. Add Base Dependencies
Import your existing requirements into the managed project. This will automatically update the `uv.lock` file.

```bash
uv add -r requirements.lock.txt
```

### 3. Install Qwen 3.5 Support (Latest Transformers)
To support the `qwen3_5` architecture, we install the cutting-edge version directly from the Hugging Face main branch:

```bash
uv add "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main" "huggingface-hub>=1.5.0"
```

### 4. Add Utility Packages (e.g., Tenacity)
Always use `uv add` to ensure the package is recorded in both `pyproject.toml` and the lock file.

```bash
uv add tenacity
```

## Daily Operations

### Syncing the Environment
Whenever you pull changes or modify the config, run `sync` to make your `.venv` match the lock file perfectly:

```bash
uv sync
```

### Upgrading Packages
To update specific core components to their latest compatible versions:

```bash
uv add --upgrade accelerate peft bitsandbytes
```

### Exporting for Legacy Systems (Optional)
If you still need a standard `requirements.txt` for Docker or other legacy tools:

```bash
uv export --no-hashes > requirements.lock.txt
```

## Troubleshooting

### Verify Model Compatibility
Run this check to ensure the `qwen3_5` architecture is registered in your environment:

```bash
python -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING; print('qwen3_5' in CONFIG_MAPPING)"
```

### Clean Reinstall
If the environment becomes corrupted, simply delete the `.venv` folder and run:
```bash
uv sync
```

