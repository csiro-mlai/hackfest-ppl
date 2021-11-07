# hackfest-probabilistic-programming

## references

* [Bayes for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
* [Rob’s course](https://robsalomone.com/course-deep-probabilistic-models/) is nice and he has given us permission to use it if we credit him. - see also https://slides.com/robsal/autodiff/#/70
* a neat [databricks example](https://databricks.com/blog/2021/06/29/using-bayesian-hierarchical-models-to-infer-the-disease-parameters-of-covid-19.html)

Cheng recommends keeping an eye on [NeurIPS tutorials](https://blog.neurips.cc/2021/06/01/neurips-2021-tutorials/), and forward-referencing them.

[Statistical Rethinking | Richard McElreath](https://xcelab.net/rm/statistical-rethinking/) has gone viral as an introduction to some of this stuff.
It is [available on O’Reilly](https://learning.oreilly.com/library/view/statistical-rethinking-2nd/9780429639142/) (free for CSIRO people).
There is a 
[PyMC3](https://github.com/gbosquechacon/statrethink_course_in_pymc3)
[and a numpyro](https://github.com/asuagar/statrethink-course-in-numpyro/)
version.

Further references:

* [that ppl guide](https://arxiv.org/abs/1809.10756) that Cheng found.

## Setup

Local:

```bash
python3 -m venv --prompt hackfest-ppl ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

HPC (not yet tested)

```bash
module add pytorch/1.9.0-py39-cuda112-mpi torchvision/0.10.0-py39
python3 -m venv --prompt hackfest-ppl --system-site-packages ./venv
pip install -r requirements.txt
```



## links discussed

* [Kalman Filter — Pyro Tutorials 1.7.0 documentation](http://pyro.ai/examples/ekf.html)
* [Example: Sequential Monte Carlo Filtering — Pyro Tutorials 1.7.0 documentation](http://pyro.ai/examples/smcfilter.html)
* [Example: Inference with Markov Chain Monte Carlo — Pyro Tutorials 1.7.0 documentation](http://pyro.ai/examples/mcmc.html)
* [Example: analyzing baseball stats with MCMC — Pyro Tutorials 1.7.0 documentation](http://pyro.ai/examples/baseball.html)
* [Mini-Pyro — Pyro Tutorials 1.7.0 documentation](http://pyro.ai/examples/minipyro.html)
* [Example: analyzing baseball stats with MCMC — Pyro Tutorials 1.7.0 documentation](http://pyro.ai/examples/baseball.html)
* [Example: Inference with Markov Chain Monte Carlo — Pyro Tutorials 1.7.0 documentation](http://pyro.ai/examples/mcmc.html)
* [Example: Sequential Monte Carlo Filtering — Pyro Tutorials 1.7.0 documentation](http://pyro.ai/examples/smcfilter.html)
* [Example: hierarchical mixed-effect hidden Markov models — Pyro Tutorials 1.7.0 documentation](http://pyro.ai/examples/mixed_hmm.html)