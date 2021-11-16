# hackfest-ppl

Authors:
- Tom Blau
- [Dan MacKinlay](http://danmackinlay.name)
- Abdelwahed Khamis
- …

Welcome to the CSIRO MLAI-FSP probabilistic programming hackfest notes.

## Schedule

Date: 22-23/11/2021

…

## Install dependencies

This step should work for everyone:

```bash
git clone https://github.com/csiro-mlai/hackfest-ppl
cd hackfest-ppl
```

Now, install the requirements.
Local desktop (not tested on windows)

```bash
python3 -m venv --prompt hackfest-ppl ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

HPC (not yet tested) in the IM&T-recommended configuration:

```bash
module add pytorch/1.10.0-py39-cuda112  torchvision/0.10.0-py39 hdf5/1.12.0-mpi
python3 -m venv --prompt hackfest-ppl --system-site-packages ./venv
pip install -r requirements.txt
```

OK, you are good to go!
## Now what?

Various notebooks walk you through different stages of the hackfest.

```text
probabilistic_programming_background.ipynb — introductory material
primitives/ — tutorial on basic operations in pyron
operator_inversion/ — advanced example using a neural network
…
```