## Running Training (without Luigi)
After cloning this repo into local computer, please checkout to the branch `dc-test`.
```
git checkout dc-test
```
Then update the submodule.
```
git submodule update --init --recursive
```
### Install poetry
If the conda is used, please install [poetry](https://python-poetry.org/docs/) under the `base` environment.
```
conda activate base
pip install pipx
pipx install poetry
poetry install .
```

### Set visible GPU and run the training
All of the parameters needed for setting up the dataset, training and logging can be found in `config-test.yaml` and `try_train.py`
```
export CUDA_VISIBLE_DEVICES=0
poetry run python run-train.py
```

