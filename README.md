# 🎰🎰🎰 hackfest-ppl 🎰🎰🎰

Authors:
- Tom Blau
- [Dan MacKinlay](http://danmackinlay.name)
- Abdelwahed Khamis
- …

Welcome to the CSIRO MLAI-FSP probabilistic programming hackfest notes!
Here you will learn how to combine modern tools of neural networks and deep learning etc, with (approximate) Bayesian reasoning, uncertainty analysis and realted techniques.
Does this solve all the problems? No.
But we argue that this provides access to more of on the pareto front trading off flexibility and computational efficiency.

Advanced case studies include experiment design, partial differential equations and whatever else you wish to bring to the party.

## Schedule

Date: 22-23/11/2021

🚧🚧🚧

## Install dependencies

This step should work for everyone:

```bash
git clone https://github.com/csiro-mlai/hackfest-ppl
cd hackfest-ppl
```

Now, install the requirements.
Local desktop.
```bash
python3 -m venv --prompt hackfest-ppl ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

This shoudl work for Linux, macos, or Windows Susbystem for Linux. 
For windows native, you are on your own good luck.

If you wish to additionally visualize graphical models, you need graphviz.
Depending on your platform this will be something like

```bash
brew install graphviz  # MacOS with homebrew
conda install graphviz # anaconda
apt install graphviz   # Debian/ubuntu/WSL default
# etc
```

[Windows is complicated](https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224) so once again, use WSL.

HPC in the IM&T-recommended configuration:

```bash
module add pytorch/1.10.0-py39-cuda112  torchvision/0.10.0-py39 hdf5/1.12.0-mpi graphviz
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
