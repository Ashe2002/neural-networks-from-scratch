# Neural Networks from Scratch

Feed-forward and recurrent neural networks implemented directly with NumPy to
demonstrate the mathematics behind forward propagation, backpropagation, and
sequence learning.

## Overview

This repository contains two neural networks built without TensorFlow, PyTorch,
Keras, or another automatic-differentiation framework. Both models predict the
next day's maximum temperature from historical weather data.

The purpose of the project is not to outperform mature machine-learning
libraries. It is to implement the core operations explicitly and develop a
deeper understanding of how neural networks learn.

## Implementations

### Feed-Forward Neural Network

`Neural_Network.ipynb` develops a fully connected regression network from first
principles.

Architecture:

```text
3 inputs -> 10 hidden units -> 10 hidden units -> 1 output
```

The notebook implements:

- parameter initialisation;
- matrix-based forward propagation;
- ReLU activation functions;
- mean squared error;
- manual gradient calculation;
- backpropagation through each layer;
- mini-batch gradient descent;
- chronological training, validation, and test sets.

The recorded run achieves a final test MSE of approximately **22.07**.

### Recurrent Neural Network

`Recurrrent_Neural_Network.ipynb` extends the project to sequential data by
implementing a recurrent neural network with:

- three input features;
- four recurrent hidden units;
- one regression output;
- seven-day input sequences;
- tanh hidden-state activation;
- manually implemented backpropagation through time;
- recurrent, input, output, and bias parameter updates.

The recurrent hidden state allows information from earlier observations in a
sequence to influence later predictions.

## Dataset

`clean_weather.csv` contains 13,509 daily observations from 1970 to 2022.

| Variable | Description |
|---|---|
| `tmax` | Maximum temperature on the current day |
| `tmin` | Minimum temperature on the current day |
| `rain` | Rainfall on the current day |
| `tmax_tomorrow` | Maximum temperature on the following day |

Missing observations are forward-filled, and predictor variables are
standardised before training. The data are divided chronologically into 70%
training, 15% validation, and 15% test sets.

## Mathematics

For a fully connected layer, the forward pass is

$$
Z^{(l)} = A^{(l-1)}W^{(l)} + b^{(l)},
$$

followed by a ReLU activation in each hidden layer:

$$
A^{(l)} = \max(0, Z^{(l)}).
$$

The recurrent network updates its hidden state using

$$
h_t = \tanh(x_tW_{xh} + h_{t-1}W_{hh} + b_h),
$$

and produces the prediction

$$
\hat{y}_t = h_tW_{hy} + b_y.
$$

Both models minimise mean squared error using gradients derived and applied
manually in NumPy.

## Repository Contents

| File | Description |
|---|---|
| `Neural_Network.ipynb` | Feed-forward network, backpropagation, training, and evaluation |
| `Recurrrent_Neural_Network.ipynb` | Recurrent network and backpropagation through time |
| `clean_weather.csv` | Daily weather dataset used by both models |

## Running the Project

Install the required packages:

```bash
pip install numpy pandas scikit-learn jupyter
```

Clone the repository and start Jupyter:

```bash
git clone https://github.com/Ashe2002/neural-networks-from-scratch.git
cd neural-networks-from-scratch
jupyter notebook
```

Open either notebook and run the cells in order. Keep `clean_weather.csv` in the
repository root so that the existing relative file path works.

## Key Skills Demonstrated

Neural-network fundamentals, linear algebra, multivariable calculus,
backpropagation, backpropagation through time, gradient descent, time-series
regression, NumPy, pandas, and model evaluation.


## Acknowledgement

The project was developed as a practical study of neural-network mechanics,
following publicly available educational material and then implementing and
annotating each stage directly.
