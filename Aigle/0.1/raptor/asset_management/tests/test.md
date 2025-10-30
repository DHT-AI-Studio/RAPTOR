# Running Tests

This document outlines the steps required to run the tests for the `asset_management` module.

## Prerequisites

Make sure you have the necessary dependencies installed before running the tests.

1. **Activate the Conda Environment**  
   Ensure you're in the correct conda environment where the required dependencies are installed.

```bash
conda activate test
```

2. **Install Dependencies**
   If you haven't already installed the required dependencies, run the following command to install them:

```bash
pip install -r requirements.txt
```

3. **Install `pytest`**
   If `pytest` is not installed in your environment, you can install it manually with the following command:

```bash
pip install pytest
```

## Running the Tests

Make sure you are in the root directory of the `asset_management` module, not inside the `tests` folder.

Once the environment is activated and dependencies are installed, you can run the tests using the following command:

```bash
PYTHONPATH=. pytest tests/
```

This command will run all the tests in the `tests` folder.

### Important:

* **Make sure to execute the test command from the parent directory of the `tests` folder**. Running the command from inside the `tests` folder may cause issues with relative imports.
* You should see the test results printed in the terminal after the command executes.

