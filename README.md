# 🎰🎰🎰 hackfest-ppl 🎰🎰🎰

Welcome to the CSIRO MLAI-FSP probabilistic programming hackfest notes!
Here you will learn how to combine modern tools of neural networks and deep learning etc, with (approximate) Bayesian reasoning, uncertainty analysis and realted techniques.
Does this solve all the problems? No.
But we argue that this provides access to more of on the pareto front trading off flexibility and computational efficiency.

Advanced case studies include experiment design, partial differential equations and whatever else you wish to bring to the party.

![](operator_inversion/fno_forward_predict_sheet.jpg)


## Now what?

Various notebooks walk you through different stages of the hackfest.

```text
probabilistic_programming_background.ipynb — introductory material
primitives/ — tutorial on basic operations in pyro
operator_inversion/ — advanced example using a neural network
…
```

## Install dependencies

This step should work for everyone:

```bash
git clone https://github.com/csiro-mlai/hackfest-ppl
cd hackfest-ppl
```

Now, install the requirements.
Local desktop:

```bash
python3 -m venv --prompt hackfest-ppl ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

This should work for Linux, macos, or Windows Susbystem for Linux. 
For windows native, you are on your own good luck.

If you wish to additionally visualize graphical models, you need graphviz.
Depending on your platform this will be something like

```bash
brew install graphviz       # MacOS with homebrew
conda install graphviz      # anaconda
sudo apt install graphviz   # Debian/ubuntu/WSL default
# etc
```

[Graphviz on Windows is complicated](https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224) so once again, use WSL.

### Bonus: HPC setup

HPC in the IM&T-recommended configuration:

```bash
module add pytorch/1.10.0-py39-cuda112  torchvision/0.10.0-py39 hdf5/1.12.0-mpi graphviz
python3 -m venv --prompt hackfest-ppl --system-site-packages ./venv
pip install -r requirements.txt
```

That did not work for me (could not lead torch), so did a brute-force upgrade

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Developer setup

If you want to contribute back tot his repository, please do.
To keep the storage small(er) we strip out all the notebooks using [nbstripout](https://github.com/kynan/nbstripout):

```bash
nbstripout --install --attributes .gitattributes
```

## Authors

- Tom Blau
- [Dan MacKinlay](http://danmackinlay.name)

With input from

- Abdelwahed Khamis
- Xuhui Fan
