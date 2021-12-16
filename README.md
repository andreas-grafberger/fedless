FedLess
================================

![CI](https://github.com/andreas-grafberger/fedless/workflows/Lint,Test,Deploy/badge.svg)

This is the source code for the paper "FedLess: Secure and Scalable Federated Learning Using Serverless Computing", 
presented at IEEE BigData 2021. The preprint can be found on [arXiv](https://arxiv.org/abs/2111.03396).

## Installation

Requires Python >= 3.7

```bash
# (Optional) Create and activate virtual environment
virtualenv .venv
source .venv/bin/activate

# Install development dependencies
pip install ".[dev]"

# Run unit and integration tests
pytest && pytest -m integ
```
