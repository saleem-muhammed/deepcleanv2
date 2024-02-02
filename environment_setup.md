# Setup python environment on LDG

[Poetry](https://python-poetry.org/docs/) will be used to set the virtual environment of python and installing the packages used by DeepClean. On LDG, we will need to install Poetry under the base environment of conda and this conda should be the conda under `$HOME/miniconda3/bin/conda` if you use [miniconda](https://docs.conda.io/projects/miniconda/en/latest/). Or you can use [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), the conda written in C++. You can use mamba under `$HOME/miniforge3/bin/mamba` to manage the virtual environments created by conda or mamba. Here are the steps to install mamba(conda) and install poetry under base:

- Install Mamba:

  1. Download the installer:

  ```bash
  $ wget [https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh)
  ```

  2. Run the installer to install Mamba:

  ```bash
  $ bash Miniforge3-Linux-x86_64.sh
  ```

  and mamba will be installed under `$HOME/miniforge3/` by default. You run

  ```bash
  $ $HOME/miniforge3/bin/mamba init
  ```

  to setup mamba in `$HOME/.bashrc`. 3. Activate base environment:

  ```bash
  $ mamba activate
  ```

- Install Poetry:
  1. Install python3.10
  ```bash
  (base) $ mamba install python==3.10.13
  ```
  2. Install [pipx](https://github.com/pypa/pipx):
  ```bash
  (base) $ pip install pipx
  ```
  3. Install poetry:
  ```bash
  (base) $ pipx install poetry
  ```
  4. Check installation of poetry:
  ```bash
  (base) $ poetry --version && which poetry
  ```
  If poetry is installed correctly, the output will be
  ```bash
  Poetry (version 1.7.1)
  ~/.local/bin/poetry
  ```
