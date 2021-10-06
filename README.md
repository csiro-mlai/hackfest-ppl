# hackfest-probabilistic-programming

## references

* https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
* https://robsalomone.com/course-deep-probabilistic-models/
* https://databricks.com/blog/2021/06/29/using-bayesian-hierarchical-models-to-infer-the-disease-parameters-of-covid-19.html

Cheng recommends keeping an eye on NeurIPS tutorials, and forward referencing them. See
https://blog.neurips.cc/2021/06/01/neurips-2021-tutorials/

## setup

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