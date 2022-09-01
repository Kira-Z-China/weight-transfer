# weight-transfer

**NOTE: This repository does not seem to yield the correct output anymore with the latest versions of Keras and PyTorch. Take care to verify the results or use an alternative method for conversion.**

This repository contains utilities for **converting Keras models to PyTorch, converting TF models to Keras, and converting TF models to PyTorch**.


## Installation
Clone this repository, and simply run

```
pip install -r requirements.txt
```

You need to have Anaconda, PyTorch, and Tensorflow installed beforehand, see the [Anaconda website](https://www.anaconda.com), [PyTorch website](https://www.pytorch.org), and the [Tensorflow website](https://tensorflow.google.cn) for how to easily install that.

## Tests

To run the unit and integration tests:

```
python setup.py test
# OR, if you have nose2 installed,
nose2
```

There is also Travis CI which will automatically build every commit, see the button at the top of the readme. You can test the direction of weight transfer individually using the `TEST_TRANSFER_DIRECTION` environment variable, see `.travis.yml`.

## How to use

See [**example.ipynb**](example.ipynb) for a small tutorial on how to use this library.

## Code guidelines

* This repository is fully PEP8 compliant, I recommend `flake8`.
* It works for both Python 2 and 3.
