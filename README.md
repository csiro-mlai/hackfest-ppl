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

## setup

For something different, we’ve set up the dependencies using [poetry](https://python-poetry.org/docs/master/basic-usage/):

```bash
## macos/linux
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -
```

```powershell
## windows
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py -UseBasicParsing).Content | python -
```

```bash
poetry install
poetry shell
```