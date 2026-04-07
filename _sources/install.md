# Installation instructions


## Clone the repository

```bash
git clone https://github.com/NatLabRockies/herc_analysis
cd herc_analysis
```

## Installation

Typical installation is to install into same environment as Hercules is installed in.

### UV Based Installation (Basic):

Can use `uv sync` to install all dependencies and Hercules:

```bash
uv sync
```

or create a new environment and install:

```bash
uv venv
uv pip install -e .
```

### UV Based Installation (Developer):

```bash
uv sync --all-extras
uv run pre-commit install
```

or create a new environment and install:

```bash
uv venv
uv pip install -e .[docs,develop]
```


### Pip Based Installation:
```bash
pip install -e .
```

### Pip Based Installation (Developer):
```bash
pip install -e .[docs,develop]
pre-commit install
```
