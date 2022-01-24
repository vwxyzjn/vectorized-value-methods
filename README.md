# Vectorized Architecture for Value-based Methods

This repository contains some early experiments combining vectorized environments and value-based methods. See [this Google Doc](https://docs.google.com/document/d/1hdRAXtqNunmcyvULkCnDOg56UfnOFYq2KqX9lSRVfyM/edit?usp=sharing)


## Get started

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)

Install dependencies:
```
poetry install
```

Train agents:
```
poetry run python vdqn.py
```

Train agents with experiment tracking:
```
poetry run python vdqn.py --track --capture-video
```

### Atari
Install dependencies:
```
poetry install -E atari
```
Train agents:
```
poetry run python vdqn_atari.py
poetry run python vc51_atari.py --track --capture-video
```
Train agents with experiment tracking:
```
poetry run python vdqn_atari.py
poetry run python vc51_atari.py --track --capture-video
```


### Pybullet
Install dependencies:
```
poetry install -E pybullet
```
Train agents:
```
poetry run python vddpg_continuous_action.py
poetry run python vtd3_continuous_action.py
```
Train agents with experiment tracking:
```
poetry run python vddpg_continuous_action.py --track --capture-video
poetry run python vtd3_continuous_action.py  --track --capture-video
```
