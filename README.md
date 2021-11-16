# hackfest-ppl

## Setup

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

HPC (not yet tested)

```bash
module add pytorch/1.10.0-py39-cuda112  torchvision/0.10.0-py39 hdf5/1.12.0-mpi
python3 -m venv --prompt hackfest-ppl --system-site-packages ./venv
pip install -r requirements.txt
```

## Now what?

Run the notebooks.