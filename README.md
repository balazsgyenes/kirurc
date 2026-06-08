# kirurc

Source code for RAL paper "Point Cloud Segmentation for Autonomous Clip Positioning in Laparoscopic Cholecystectomy on a Phantom".

# Installation

Install [Blender](https://www.blender.org/) version 3.4:

```bash
sudo snap install blender --channel=3.4/stable --classic
```

Create a conda environment with python 3.10.

```bash
mamba create --prefix ./.env python=3.10
mamba activate ./.env
```

Install openexr and openexr-python with mamba, since the required package versions don't exist on pypi:

```bash
# the versions we require don't exist on pypi
mamba install openexr=3.1 openexr-python=1.3.9
```

Install torch and torch geometric.
We mandate numpy 1.\* rather than numpy 2.\*.
When using numpy 2.\*, open3d's `utility.Vector3dVector` triggers a segfault if used with `np.float32` dtype.

```bash
pip install torch torch_geometric "numpy<2.0"

TORCH_PLUS_CUDA=$(python -c "import torch; print(torch.__version__)")
# on the cluster, first load nvcc
# module load devel/cuda/12.4
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-$TORCH_PLUS_CUDA.html
```

Install remaining requirements using pip:

```bash
pip install -r requirements.txt
```

`pc-skeletor` hard-codes a minimum point cloud size that we need to reduce.
Open `$CONDA_PREFIX/lib/python3.10/site-packages/pc_skeletor/skeletor.py:L164` and replace `n_neighbors=30` with `n_neighbors=10`.

Install kirurc_vision repo itself as editable install:

```bash
pip install -e .
```

## Stratified Transformer (from Pointcept)

Install additional dependencies using pip.
We don't install `torch-points3d` because it has terrible code structure and many dependencies it doesn't actually need.
We just install the cuda kernels:

```bash
pip install numba timm

# on the cluster, first load gcc 11
# module load compiler/gnu/11
# module load devel/cuda/12.4
pip install --no-deps --no-build-isolation torch-points-kernels
```

Clone the Pointcept repo and compile the required cuda kernels:

```bash
git clone git@github.com:Pointcept/Pointcept.git
cd Pointcept/libs/pointops2
python setup.py install
```

## Fast Point Transformer

```bash
mamba create -p .env2 python=3.8
mamba install openexr=3.1 openexr-python=1.3.9

# conda install does not work for torch 1.10
pip install "numpy<2.0" torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install torch_geometric==2.0.4
pip install torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
pip install -r requirements.txt
pip install -e .

cd ${FASTPOINTTRANSFORMER}

cd thirdparty/MinkowskiEngine/
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas --force_cuda
cd ../..

cd src/cuda_ops
pip3 install .
cd ../..

pip install pytorch-lightning lightning-bolts wrapt gin-config rich einops

```

# Usage

Generate synthetic data using
```bash
python scripts/generate_synthetic.py
```

Start pretraining using
```bash
python pretrain.py
```

Start fine-tuning using
```bash
python fine_tune.py model_artifact=[ARTIFACT_NAME]
```
where `ARTIFACT_NAME` is the name of the wandb artifact of the pretrained model.

We use hydra for configuration, which can be found in `conf/`.
Useful scripts, e.g. for visualizing data and evaluation, can be found in `scripts/`.


# Code Architecture

## Regression Ablation

As an ablation, we modify the network to output the coordinates of the 6 target points directly (see `conf/pretrain_experiment/regression.yaml`).
For this, the classification version of PointTransformer is used instead of the segmentation version, with the same (cosmetic) changes to the code.
As a preprocessing step, each reference spline is evaluated at 35%, 50%, and 65% to generate the 6 target points (see `kirurc_vision/transforms/add_target_points.py`).
In order to ensure that the data augmentation transforms are propagated to the target points, they are added to the point cloud itself (`pos` field).
This takes care of the RandomRotate, RandomStretch, and RandomShear transforms, which should also apply to the targets.
Cutout (see `kirurc_vision/transforms/cutout.py`) and jitter (see `kirurc_vision/transforms/jitter.py`), which are both destructive operations, are modified to have no effect on the target points.
After data augmentation, the target points are moved to the `y` field of the data (see `kirurc_vision/transforms/add_target_points.py`), since only this field is available to the loss function.
