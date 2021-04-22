import numpy as np
import os
import pandas as pd
import pymc3 as pm
import seaborn as sns

from matplotlib import pyplot as plt
from pymc3.distributions.timeseries import GaussianRandomWalk
from theano import tensor as T

###
os.system("Rscript -e 'require(HSAUR); data(\"mastectomy\", package=\"HSAUR\"); "
          "write.csv(mastectomy,\"./ignoreland/mst.csv\")'")
os.listdir("./ignoreland")

df = pd.read_csv("./ignoreland/mst.csv")
df.info()
df["event"] = df["event"].astype("int")
df["met"] = df["metastized"].map({"yes":1,"no":0}).astype("int")
df.groupby("met").count()
n_patients = len(df)
patients = np.arange(n_patients)  # create nd array

df["event"].mean()  # 60% of subjects had endpoint; remainder censored
fig, ax = plt.subplots(figsize = (8, 6))
blue, _, red = sns.color_palette()[:3]
ax.hlines(
    patients[df.event.values == 0], 0, df[df.event.values == 0].time,
    color=blue,
    label="Censored"
)
ax.hlines(
    patients[df.event.values == 1], 0, df[df.event.values == 1].time,
    color=red,
    label="Uncensored"
)
ax.scatter(
    df[df.met.values == 1].time,
    patients[df.met.values == 1],
    color="k",
    zorder=10,
    label="Metastasized",
)
ax.set_xlim(left=0)
ax.set_xlabel("Months since surgery")
ax.set_yticks([])
ax.set_ylabel("bc_subject")
ax.set_ylim(-0.25, n_patients + 0.25)
ax.legend(loc="center right")
# Setting up priors:
interval_length = 3
interval_bounds = np.arange(0, df["time"].max() + interval_length + 1, interval_length )
n_intervals = len(interval_bounds) - 1
intervals = np.arange(n_intervals)
# Plot distribution of time to events (censored and uncensored)
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(
    df[df.event == 1].time.values,
    bins=interval_bounds,
    color=red,
    alpha=0.4,
    lw=0,
    label="Uncensored",
)
ax.hist(
    df[df.event == 0].time.values,
    bins=interval_bounds,
    color=blue,
    alpha=0.4,
    lw=1,
    label="Censored",
)

ax.set_xlim(0, interval_bounds[-1])
ax.set_xlabel("Months since mastectomy")

ax.set_yticks([0, 1, 2, 3])
ax.set_ylabel("Number of observations")

ax.legend();

last_period = np.floor( (df["time"] - 0.01) / interval_length).astype("int")
## initialize death count matrix, 44 by 76
death = np.zeros( (n_patients, n_intervals) )
# populate
death[patients, last_period] = df["event"]  # by array index
death[16, 5]  # the ith patient's last period
# use the outer product implementation of ge to get a matrix of true/false for whether a person
# survived longer than the interval; multiply through by a constant (since each period represents 3 units).
# Must convert pd series to numpy to get this to work.
exposure = np.greater_equal.outer(df["time"].to_numpy(), interval_bounds[:-1]) * interval_length
# truncate the last period's exposure time (if it is not a full 3 units)
exposure[patients, last_period] = df["time"] - interval_bounds[last_period]

SEED = 644567  # reproducing the results from the abovementioned blog post!
with pm.Model() as model:
    lambda0 = pm.Gamma("lambda0", 0.01, 0.01, shape=n_intervals)  # initialize this
    beta = pm.Normal("beta", 0, sigma=1000)  #

    lambda_ = pm.Deterministic("lambda_", T.outer(T.exp(beta * df["met"]), lambda0))
    mu = pm.Deterministic("mu", exposure * lambda_)
    obs = pm.Poisson("obs", mu, observed=death)

n_samples = 2000
n_tune = 2000
with model:
    trace = pm.sample(n_samples, tune=n_tune, random_seed=SEED)

trace["beta"].mean()
np.exp(trace["beta"].mean())
pm.plot_posterior(trace, var_names = ["beta"], color="#87caab")


