# DeepClean, take 3
Here's a restructure of DeepClean intended for at least a few purposes:
- Taking advantage of [lightning](https://lightning.ai/) during training for faster, more modular, and simpler training code
- Containerizing DeepClean projects for easier uptake by new users
- Serving as a proof of concept and sandbox for a `law`-based workflow with custom configs
- Serving as a proof of concept for use of a centralized [`toolbox`](./toolbox/) containing standard ML4GW libraries and code-styling configs
- Generalizing many of the components of DeepClean for easier experimentation with new architectures and for target new frequency bands with the [`couplings`](./deepclean/couplings) submodule of the pipeline library

I'll expand more on these when I have time, but for now I'll add a couple instructions to get started with training.

## Run instructions
### Environment setup
| **TODO: data authentication instructions**

#### Install
Before attempting to build anything, ensure that you have the relevant submodules initialized

```bash
git submodule update --init --recursive
```

##### `pip` instructions: not recommended
You can install the local library via `pip`:

```bash
python -m pip install -e .
```

##### `poetry` instructions: recommended
However, I'd recommend using some sort of virtualization software. This repo is automatically compatible with [`poetry`](https://python-poetry.org/), my personal favorite. In that case, you would just need to do

```bash
poetry install
```

#### Directory setup
Set up a local directory to save container images that we'll export as `DEEPCLEAN_CONTAINER_ROOT`.
**IMPORTANT: This directory should _not_ live locally with the rest of this code. Apptainer has no syntax for excluding files from being added into containers at build time, so if you save your (often GB-sized) containers here, you'll make building the next container enormously more painful. This means that you'll need to build your images on every _filesystem_ (not node) that you intend to run on. In the future, we'll make these containers available on `/cvmfs` via the Open Science Grid, but for now you'll need to build them**.

```bash
# or wherever you want to save this
export DEEPCLEAN_CONTAINER_ROOT=~/images/deepclean
mkdir -p $DEEPCLEAN_CONTAINER_ROOT
```

Finally make a directory to save our data and run outputs

```bash
export DATA_DIR=~/deepclean/data
mkdir -p $DATA_DIR

export RESULTS_DIR=~/deepclean/results
mkdir -p $RESULTS_DIR
```

### Configuring your cleaning problem
DeepClean can be easily configured to remove any number of different noise couplings from the strain data, and targeted at any interferometer for which that coupling has had witness channels identified in the [`couplings`](./deepclean/couplings) submodule.
To run a pipeline for a particular coupling, just set the environment variable `DEEPCLEAN_IFO` to the inteferometer you want to target, e.g. `export DEEPCLEAN_IFO=H1`, then edit the `problem` entry in the `deepclean` table of your [`luigi.cfg`](./luigi.cfg) to specify the comma-separated list of couplings to target.
Right now, only the channels for the `60Hz` coupling at Hanford and Livingston (`H1` and `L1`) are defined, but this structure should make it simple to train on new couplings when their corresponding witnesses are identified.

### Dataset generation
Start by building the data generation container

```bash
cd projects/data
apptainer build $DEEPCLEAN_CONTAINER_ROOT/data.sif apptainer.def
cd -
```

Then you can query a stretch of time for all active segments and download the data for those segments via:

```bash
poetry run law run deepclean.tasks.Fetch \
    --data-dir $DATA_DIR \
    --start 1250899218 \
    --end 1252108818 \
    --sample-rate 4096  \
    --min-duration 8192 \
    --max-duration 32768 \
    --image data.sif \
    --job-log fetch.log
```

### Training
Once you've generated your training data, you're ready to train! Start by building your training container image

```bash
cd projects/train
apptainer build $DEEPCLEAN_CONTAINER_ROOT/train.sif apptainer.def
cd -
```

Find a node with some decently-sized GPUs, ensure that the one you want isn't being used, and then run (assuming you built this library with `poetry`):

```bash
export GPU_INDEX=0  # or whichever you want
poetry run law run deepclean.tasks.Train  \
    --image train.sif \
    --gpus $GPU_INDEX \
    --data-fname $DATA_DIR/deepclean-1250916945-35002.hdf5 \
    --output-dir $RESULTS_DIR/my-first-run
```

#### Making changes to the code
If you make changes to the code that you want to experiment with, you can map them into the container at run time without having to rebuild simply by passing the `--dev` flag to any of the `law run` commands.
