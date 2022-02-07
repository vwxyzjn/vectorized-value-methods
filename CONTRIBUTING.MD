# Contribution Guide

👍🎉 Thank you for taking the time to contribute! 🎉👍

To get an overview of some of this project's motivation and goals, you can take a look at [this Google Doc](https://docs.google.com/document/d/1hdRAXtqNunmcyvULkCnDOg56UfnOFYq2KqX9lSRVfyM/edit?usp=sharing).
Feel free to open an issue or pull request if you have any questions or suggestions.
You can also [join our Discord](https://discord.gg/Ut87avZGuT) and ask questions there.
If you plan to work on an issue, let us know in the issue thread so we can avoid duplicate work.

## Dev Setup

```bash
poetry install
poetry install -E atari
poetry install -E pybullet
```

Then you can run the scripts under the poetry environment in two ways: `poetry run` or `poetry shell`. 

* `poetry run`:
    By prefixing `poetry run`, your command will run in poetry's virtual environment. For example, try running
    ```bash
    poetry run python vdqn.py
    ```
* `poetry shell`:
    First, activate the poetry's virtual environment by executing `poetry shell`. Then, the name of the poetry's
    virtual environment (e.g. `(vectorized-value-methods-T4LxsEjE-py3.9)`) should appear in the left side of your shell.
    Afterwards, you can directly run
    ```bash
    python python vdqn.py
    ```


## Code Formatting

We use [Pre-commit](https://pre-commit.com/) for code formatting, sorting inputs and other utilities. You can run the following command:

```bash
pre-commit run --all-files
```

## How to run hyper-parameter sweeps

Create a `sweep.yaml` file like the following

```yaml
command:
- ${env}
- ${interpreter}
- ${program}
- --track
- ${args}
method: bayes
metric:
  goal: maximize
  name: charts/episodic_return
parameters:
  num-minibatches:
    values:
    - 1
    - 2
    - 3
    - 4
  update-epochs:
    values:
    - 1
    - 2
    - 3
    - 4
  q-lr:
    values:
    - 0.001
    - 0.0003
    - 0.0001
program: vsac_continuous_action_returns.py
```

Then run 

```bash
wandb sweep -p vectorized-value-methods -e vwxyzjn sweep.yaml
# Here `-p` stands for `wandb-project-name` and `-e` the `wandb-entity`.
```

This command in turn will Initialize a sweep with an id like `o9bwxmbv` and follow the printed instruction from the command above to run something like 

```bash
wandb agent vwxyzjn/vectorized-value-methods/o9bwxmbv
```

You may run multiple `wandb agent ...` commands at the same time.

