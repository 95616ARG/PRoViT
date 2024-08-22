# Provable Repair of Vision Transformers
PRoViT is an approach that repairs Vision Transformers.
Thousands of images can be repaired using PRoViT while preserving the 
Vision Transformer architecture.

The code in this repository is the latest artifact from our paper
***Provable Repair of Vision Transformers***, published at SAIV 2024.
```
@inproceedings{SAIV2024,
  title={Provable Repair of Vision Transformers},
  author={Nawas, Stephanie and Tao, Zhe and Thakur, Aditya V},
  booktitle={International Symposium on AI Verification},
  pages={156--178},
  year={2024},
  organization={Springer}
}
```

## Installation

### Local Installation

If you wish to run PRoViT locally, the reference environment is `Linux` (`Ubuntu
20.04`) with `Python 3.9.7`, `torch 1.11.0` and `torchvision 0.12.0`. Note that
we recommend using `Python 3.9.7`, as other versions may not be
compatible with `torch 1.11.0`. We suggest using a conda environment to run the experiments. Run the following command to install required
Python packages.  

```
$ pip3 install -r requirements.txt
```

If you wish to use NVIDIA GPU/CUDA, the reference environment uses CUDA 11.3 and
CUDNN 8. You could change the following lines in `requirements.txt` to a CUDA
version that's compatible with your CUDA installation.
```
torch==1.11.0+cu113
torchvision ==0.12.0+cu113
```

## Prerequisites

### Download and Extract Datasets

Our experiments require ImageNet-C and ImageNet validation datasets. Please
download the [official ImageNet validation set
(`ILSVRC2012_img_val.tar`)](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5)
via torrent and place it in `~/datasets/ILSVRC2012/ILSVRC2012_img_val.tar`. The
following command will extract the
Imagenet validation dataset.
```
$ make datasets-imagenet
```

For ImageNet-C, download `weather.tar` from ImageNet-C's [Zenodo](https://zenodo.org/records/2235448). Extract the tar file to your datasets directory and rename the folder to `imagenet-c`.

### Set Up Gurobi License

Reproducing experiments for PRoViT requires a (free) Gurobi
academic license. Please visit [Gurobi academic
license](https://www.gurobi.com/academia/academic-program-and-licenses) to
generate an "Academic WLS License" (for containers). Aside from the official
instructions, the following steps might be helpful.

- Login to the Gurobi user portal.
- Go to the "License - Request" tab, genearte a "WLS Academic" license if you don't have
  one. If you already have a "WLS Academic" license, you might get an
  "[LICENSES_ACADEMIC_EXISTS] Cannot create academic license as other academic
  licenses already exists" error.
- Go to the "Home" tab, click "Licenses - Open the WLS manager" to open the WLS
  manager.
- In the WLS manager, you should see a license under the "Licenses" tab. Click
  "extend" if it has expired (it might take some time to take effect).
- Go to the "API Keys" tab, click the "CREATE API KEY" button to create a new
  license, download the generated `gurobi.lic` file and place it in
  `/opt/gurobi/gurobi.lic` inside the container.

### Hardware Requirements

All experiments were run on a machine with Dual Intel Xeon Silver 4216 Processor
16-Core 2.1GHz with 384 GB of memory, SSD and RTX-A6000 with 48 GB of GPU memory
running Ubuntu 20.04. Running on a machine with less CPU/GPU cores and memory
might not reproduce the timing numbers in the paper.

Running the experiments without GPU will be much slower, especially during 
fine tuning and evaluation.

## Getting Started Guide

The scripts `./run_exp{i}.sh` are set up to run the experiments from the
SAIV 2024 publication, each experiment labeled as in the paper.
If the script cannot find the datasets on your machine, you may need to edit the path option in the scripts, for example:
```
--path /home/public/datasets/ImageNet
```
Experiment logs are saved in the `./logs` directory.

> Note: Gurobi license required, see "Setup Gurobi License" for details.

```
cd PRoViT
experiments/run_exp2.sh
```

To run other experiments, run the following:

    python3 experiments/vit_repair.py --path ~/datasets --netname deit --device cuda:0 --n 10 --seed 0 --metric 3 --method provitFTLP --ft_niter 1 --batch_size 100

You can change the options for different experiments. The options are:

- path: The directory containing the ImageNet-C dataset and the ImageNet validation set.
- netname: Which network to repair (vitb16, vitl32, deit, resnet152, vgg19)
- device: (cpu, cuda, cuda:0) Note that we only tested with RTX A6000
(48GB), hence running larger experiments on GPU with less memory might cause
failure.
- n: The number of labels to include in the repair set.
- seed: An integer for the random seed to create different repair sets.
- metric: An integer representing which metric (as described in `provit/datasets.py`) to use to construct the repair and generalization sets.
- method: Which method to use for repair. The options are provitFT, FTall, provitLP, and provitFTLP.
- ft_niter: The max number of iterations of fine tuning to complete before timing out.
- batch_size: The number of inputs to group in each fine tuning update.
- lr: The learning rate for fine tuning.
- gamma: The multiplicative factor of learning rate decay in the learning rate scheduler.




Note that the timing numbers may not be the same due to the difference in
hardware. The drawdown and generalization numbers may not be exactly the same
for the following reasons:

- The Gurobi solver, especially its concurrent methods, is not deterministic.
  Hence the experiment might produce a different repaired network.
- Difference in hardware (e.g., CPU, GPU, Tensor cores), instruction sets and
  libraries (e.g., CUDA, CUDNN) might cause small differences in the evaluation
  of accuracy. 


# Troubleshooting and Frequently Asked Questions

## Why do I see `gurobipy.GurobiError: Model too large for size-limited license`?

This is because the Gurobi academic license is missing and Gurobi is using a
trial license shipped with the `gurobipy` package. Please follow the "Setup
Gurobi License" section to acquire one and put it (or paste its content to)
under `/opt/gurobi/gurobi.lic`. To verify the license, the command

```
cat /opt/gurobi/gurobi.lic
```

should print a license like

```
# Gurobi WLS license file
# Your credentials are private and should not be shared or copied to public repositories.
# Visit https://license.gurobi.com/manager/doc/overview for more information.
WLSACCESSID=<WLSACCESSID>
WLSSECRET=<WLSSECRET>
LICENSEID=<LICENSEID>
```

And you should be able to see the following lines in the console output of experiments.

```
Set parameter WLSAccessID
Set parameter WLSSecret
Set parameter LicenseID to value <LICENSEID>
Academic license - for non-commercial use only - registered to <username or email>
```

Also, after running any experiment with your license, you should be able to
login https://license.gurobi.com/manager/keys and see activities of the
corresponding license.

## Why do I see "Gurobi license expired"?

There are few possible reasons:

1. It might because you haven't put your academic license
   `/opt/gurobi/gurobi.lic` and the trial license has expired. In this case,
   please follow the "Setup Gurobi License" section to install your license.

2. It might because your Gurobi WLS license is expired. You could login
   https://license.gurobi.com/manager/licenses and check the status of your
   Gurobi WLS license. If it is expired, check `extend` to extend it.

3. It might because the Gurobi server haven't update the expiration date of your
   license if you just registered one or extended it. In this case, please wait
   for a few minutes.

## Why do I see a `ModuleNotFoundError: No module named ...` exception?

It is because the python virtual environment (venv) is deactivated in the shell.
Use `conda activate` to activate the conda environment used for these experiments.
