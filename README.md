# MLSES Seminar 23: Solar Thermal System

This seminar work is about time series forecasting and function approximation for solar thermal system data.

The solar thermal system contains four different circuits: Boiler, heating, solar, and water. 
Each circuit incorporates three temperature sensors (T1-T3) as well as a sensor for the volume flow rate (VF).
Moreover, the quantity heat transfer (Qth) is of interest, which is in a functional relation to the temperatures and the volume flow rate, i.e., can be calculated from the former.

The first task, *time series forecasting*, is to predict VF and Qth for one timestep (5s) and one hour into the future, respectively, given all sensor data of the past.
The second task, *function approximation*, is to predict VF and Qth, given only temperature sensor data of the past and the future.

---

## Prerequisites

To run the scripts in this repository, **Python 3.10** and **at least 12GB RAM** are needed.
Then, simply create a virtual environment and install the required packages via

```bash
pip install -r requirements.txt
```

The prepared data (available on Ilias) is expected to be located in the directory `./data/`, such that the directory structure is as follows:

```
data/
├── boiler/
│   ├── source_test.npy
│   ├── source_training.npy
│   ├── target_test.npy
│   └── target_training.npy
├── heating/
├── solar/
└── water/
```

Here, the subdirectories `heating/`, `solar/` and `water/` are structured analogously to `boiler/`.

---

## Model fitting and testing

For both model fitting and testing, the same python script `./main.py` is used.
It implements a command line interface that allows choosing between fitting and testing, selecting the task, and configuring other parameters.
For the details, invoke the script with the flag `-h`/`--help`.

In the following, we present the most important arguments:

### **--task** (mandatory)

Selects the task ('forecast_step', 'forecast_hour', 'approximation').

### **--test**

Activates testing mode. Requires a model being fitted for the selected task with same model parameters (`--window-size`, `--time-features`, `--hour-horizon-partitions`, see below).

### **--window-size**

Defines the size of the sliding window on the time series for input variables in the selected task.
The methodogical approach is explained in [Methods](#methods). 

### **--time-features**

Activates time features, which are then added to the input variables.
Here, we use the feature 'day in week', which is then one-hot-encoded.

### **--horizon-partition-size**

Defines the size of the horizon partitions in the 'forecast_hour' task.
Must be a factor of 720 (which is the size of the total horizon).
Again, we refer to  [Methods](#methods) for details.

---

For example, to fit a model on the 'approximation' task with default parametrization, call

```bash
python main.py --task approximation
```

After the fitting process is finished, a corresponding `.json` file is created in the directory `./models/`, which contains the model parameters.
It is automatically found and loaded, when invoking the script in testing mode afterwards.

---

## Methods

### XGBoost

We use a gradient-boosted decision tree model ([*XGBoost*](https://xgboost.readthedocs.io/en/stable/)) to predict the quantities of interest.
The utilized regression model is capable of multiple inputs and multiple outputs in a one-dimensional, unstructured fashion, i.e., it performs vector-to-vector predictions, where the order of vector components is not exploited by the model.

### Sliding Window

To make use of the information encoded in past and/or future data points relative to a specific data point, we use a sliding window approach on the input domain and predict in an autoregressive fashion.
In the following, we denote the time series of so-called *source variables* (T1, T2, T3, time features) by $\mathcal{X}: \mathbb{T} \times \mathbb{R}^{d_1}$, and *target variables* (VF) by $\mathcal{Y}: \mathbb{T} \times \mathbb{R}^{d_2}$. 

For the approximation task, the window is bidirectional and centered at the time index of interest $t$. 
Therefore, its size $\tau$ must be odd (since the time axis $\mathbb{T}$ is discrete).
To predict $\mathcal{Y}(t, :)$, the model takes the window $\mathcal{X}(t-\frac{\tau}{2} : t+\frac{\tau}{2}+1, :)$ as input, where we used numpy-notation to denote a slicing along a dimension.
As the XGBoost model only accepts one-dimensional inputs and outputs, we also have to flatten the two-dimensional window, which is not explicitly expressed here and subsequently for the sake of brevity.

For the two forecast tasks, the window is unidirectional and ends at the time index prior to the (first) time index of interest $t$.
In contrast to the 'approximation' task, the window here contains both source and target variables.
This conjunction of $\mathcal{X}$ and $\mathcal{Y}$ along the feature dimension is denoted by $(\mathcal{X} +\!\!\!+~ \mathcal{Y}): \mathbb{T} \times \mathbb{R}^{d_1+d_2}$.

Analogously, a second window (called 'horizon' to distinguish) starts at $t$.
In the 'forecast_hour' task, its size is 720 (1h / 5s = 3600s / 5s = 720), in the 'forecast_step' task, it is 1.

To predict $(\mathcal{X} +\!\!\!+~ \mathcal{Y})(t, :)$ in the 'forecast_step' task, the model takes the window $(\mathcal{X} +\!\!\!+~ \mathcal{Y})(t-\tau : t, :)$ as input.

Last, for the 'forecast_hour' task, we could theoretically extend this to the prediction of $(\mathcal{X} +\!\!\!+~ \mathcal{Y})(t : t + 720, :)$.
However, this is computationally quite demanding, and needs lots of memory.
Therefore, we split the total prediction horizon into smaller partitions, which are then iteratively predicted using the previously predicted horizon partition.
Formally, our model predicts $(\mathcal{X} +\!\!\!+~ \mathcal{Y})(t+i\lambda : t + (i+1)\lambda, :)$ using $(\mathcal{X} +\!\!\!+~ \mathcal{Y})(t + i\lambda - \tau : t + i\lambda, :)$ as input window, where $i\in\{0, 1, ..., \frac{720}{\lambda}\}$ and $\lambda\in\mathbb{N}$ refer to the current iteration and the size of the horizon partitions, respectively.
Note that $\lambda$ needs to be a factor of 720 and that $\lambda \geq \tau$ has to hold in order to make this work. 

### Data Preprocessing

Since the provided source data has a much more fine-scaled time discretization (0.1s) compared to the target data (5s), we simply average out the former over subsequences of 40 time indices.

For the computation of the 'day in week' time feature, we use [*Zeller's congruence*](https://en.wikipedia.org/wiki/Zeller%27s_congruence).

We split the provided training dataset into a training and a validation subset, with ratio 80:20.
For the forecast tasks, only a deterministic cut is allowed to avert information leakage from the future to the past.
In the 'approximation' task, however, a random split might be used, if the window size is set to 1, i.e., using only the present and no past or future information of the source variables.
The fitting process is then early stopped w.r.t. the predictive performance on the validation subset, to prevent overfitting.

---

## Results

We evaluate the predictive performance of our model on the provided test dataset, which is withheld during the fitting process.
In this study, we use the two metrics *root-mean-square-error (RMSE)* and *mean-absolute-error (MAE)*.
A prediction is created for each valid time index (i.e., such that there are sufficient time indices before and after to fill the window and the horizon).
Furthermore, we evaluate the metrics for the individual circuits, but also summing across circuits.
In the following, we only report the latter.
For full results and visualizations, we refer to the Jupyter notebook `./notebooks/results.ipynb`.

### Approximation Task

```
--random-split
--window-size 1

RMSE (VF_total) = 171.29717
MAE (VF_total) = 101.60860
RMSE (Qth_total) = 1403.11506
MAE (Qth_total) = 832.28809
```

### Step Forecast Task

```
--window-size 5

RMSE (VF_total) = 9.85822
MAE (VF_total) = 5.78839
RMSE (Qth_total) = 63.59658
MAE (Qth_total) = 13.19169
```

### Hour Forecast Task

```
--window-size 5
--horizon-partition-size 5

RMSE (VF_total) = 248.72072
MAE (VF_total) = 120.33562
RMSE (Qth_total) = 2037.29927
MAE (Qth_total) = 985.68307
```