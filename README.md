# DeepClean, take 3
Here's a restructure of DeepClean intended for at least a few purposes:
- Taking advantage of [lightning](https://lightning.ai/) during training for faster, more modular, and simpler training code
- Containerizing DeepClean projects for easier uptake by new users
- Serving as a proof of concept and sandbox for a `law`-based workflow with custom configs
- Serving as a proof of concept for use of a centralized [`toolbox`](./toolbox/) containing standard ML4GW libraries and code-styling configs
- Generalizing many of the components of DeepClean for easier experimentation with new architectures and for target new frequency bands with the [`couplings`](./deepclean/couplings) submodule of the pipeline library

I'll expand more on these when I have time, but for now I'll add a couple instructions to get started with training.

## Run instructions
**NOTE: all commands are expected to run on LDG on GPU-enabled nodes. I particularly recommend the DGX nodes at the detector sites**

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

## Where things stand
Since this code is in my local repo, someone should set up an ML4GW/deepcleanv2 repo and push this code there so that it can be managed by remaining researchers.

### Offline training/cleaning
First of all, lots of good work on scaling up training is being done in the [aframev2 repo](https://github.com/ml4gw/aframev2). This includes

- Project environments that inherit from the pipeline environment (i.e. the one here at the root level) and specify the correct law executable in the `law.cfg` in order to be able to leverage the environment's libraries in the `Task.run` method.
- Support for distributed training on the Nautilus cluster via Kubernetes
- Setting up logging and validation in distributed training settings
- Support for automated hyperparameter searching via Ray Tune
- Better S3 support for data storage and training artifact storage
- Integration with model export and inference
- Luigi/law level APIs for S3 artifacts and spinning up ray clusters

The [README](https://github.com/ml4gw/aframev2/tree/main/README.md) includes lots of good info about the two layers of the code (project level vs. pipeline level) and explains some helpful concepts for things like Weights & Biases.

On the other hand, there's some good stuff here that would be valuable to aframev2. I'm thinking in particular of wandb-bokeh logging.
There should probably start being a unified library layer that consolidates a lot of this functionality into one place.
Moreover, these two projects should be coordinating a lot to see how they're solving their infra problems.
At the infra level they're much more similar than they are different, and there's lots of work that would benefit both.

#### Full pipeline
A single run of the train project can take care of training and cleaning data offline.
To simulate the production retraining pipeline, it should be straightforward to set up a Luigi `Task` that
1. Downloads segments from a particular stretch
2. Iteratively fine-tunes a model on segments during that stretch at some fixed cadence by leveraging the `--data.start_offset` and `--data.duration` flags. Unfortunately, I don't have support for loading pre-trained weights yet, but that should be more or less trivial.
3. Potentially performs a hyperparameter search on the first training to figure out good baseline training parameters that can be used for the remaining training (with a significantly reduced learning rate to fine-tune).

This should be the default pipeline to run to evaluate the efficacy of any new ideas, which can be compared directly via the W&B dashboard.

### Online pipeline
The [microservice project](https://github.com/alecgunny/deepclean/tree/ml4gw-introduction/projects/microservice) used to implement IaaS inference needs to be deprecated for a few reasons.
The main one is that there seems to be issues with exporting the model to ONNX, which is causing the IaaS deployment to produce poorly cleaned data.
More broadly, however, the current compute requirements of DeepClean are so light as to make a full IaaS deployment unnecessary, while incurring a lot of development difficulty in managing asynchronous applications.

However, all of the implementation of how cleaning happens _around_ the IaaS deployment can be used more or less as-is, just with an in-memory Torch model.
This includes lots of logic about how the multiple data streams DeepClean produces are generated, and a lot of index tracking to know when it's time to clean the next frame (and which samples in the timeseries of noise predictions align with which samples in the timeseries of raw strain data).
The only addition this will need is the logic for loading and unloading new and stale models, respectively, functionality Triton used to handle for us.
This will require some level of communication between the offline monitor and the online cleaning code.
You could adopt a REST API-like implementation like the microservice does, but it might be simpler to communicate via checking for new files.

The training implementation from `microservice` could remain as-is as well, but given that
1) The training is functionally offline
2) The `deepclean` pipeline makes it simple to launch `luigi.Task`s for downloading data offline and training on it
3) All `projects` should inherit `deepclean` as a dependency and can therefore launch `luigi.Task`s

I personally think it makes sense to ditch the live-frame training code and just launch a small fetch-then-train pipeline from the online cleaning code at a fixed cadence, but it's up to the next set of developers how they want to handle this.

One key thing is that the offline validation and cleaning pipelines try to implement the online cleaning scenario as much as possible, i.e. they
- Clean in 1s increments
- Only use a small amount of future and past data to perform bandpass filtering
- Throw away predictions near the edges of kernels

However, the do these tasks in an online fashion to be as efficient as possible.
The online cleaning code will just need to be implemented differently (and again, the existing microservice cleaning code is correct), but there should be automated testing that ensures that the cleaned frames produced by these two pipelines match up to precision differences.

### What's next
- The `deepclean.couplings` API makes it trivial to specify new subtraction problems as long as you have some relevant witness channels and frequnency range. You can even attack multiple ranges at once by specifying `problem = <comma separated list of subtraction problem names>` in [`luigi.cfg`](./luigi.cfg)
- We should be using much bigger models, especially for lower-frequency noise where we'll need a broader field-of-view in order for a prediction to "observe" more cycles of the relevant frequency components.
    - This should take advantage with the new `autoencoder` library added to `ml4gw` in order to specify arbitrarily large autoencoders with useful tricks like skip-connections.
    - Eventually transformers will make the most sense to use, but those are much more unwieldy to train and frankly don't seem like low-hanging fruit (which is always better data).
- The code for building saliency maps and calculating coherence for analyzing the contribution of individual channels to model learning is [here](https://github.com/ML4GW/DeepClean/blob/main/libs/trainer/deepclean/trainer/analysis.py), and the code for plotting it is [here](https://github.com/ML4GW/DeepClean/blob/ed9418c06460cc9de437e4a2a47be77da9c20cbf/libs/viz/deepclean/viz/plots.py#L61)
    - This should be included as a post-training analysis step (as it was in the old repo) and plotted to W&B. It could be used with a hyperparameter search as a means for identifying important witnesses from a large pool of candidates
    - This could even be its own pipeline, which doesn't need retraining since its meant to be primarily exploratory on short stretches of data where a noise source is known to be present. It would instead optimize for larger hyperparameter searches to make sure that any potential signal is identified.
    - An interesting research topic with the 60Hz noise would be to remove those channels that are shown to have low saliency, even though they have high coherence, and see if the same performance can be recovered. That would be a powerful way of proving the efficacy of this search method.
