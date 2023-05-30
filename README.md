# sia-tp4

## Dependencies

- Python 3.10

## Usage

```shell
pipenv install
```

For each exercise (kohonen, oja and hopfield) the project can be run as following:

```shell
pipenv run python run_X.py X.json
```

## Configuration files

### Kohonen

The configurable parameters are the learning rate and radius, and whether they are constant or variable also the size of
the grid and how to initialize the weights (**"data"** meaning to use weights from the provided data and **"random"** to
use random values)

```json
{
  "learning_rate": 0.1,
  "constant_lr": true,
  "radius": 1.5,
  "constant_radius": true,
  "size": 4,
  "init_weights": "random"
}
```

### Oja

```json
{
  "learning_rate": 0.001
}
```