import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import theano as tno

from pymc3 import Model, Normal, Slice, sample, traceplot
from pymc3.distributions import Interpolated
from scipy import stats

####
# Following http://docs.pymc.io/pymc-examples/examples/pymc3_howto/updating_priors.html

np.random.seed(867)
# True values
alpha_true = 5
beta0_true = 7
beta1_true = 13

size = 100

X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.25

Y = alpha_true + beta0_true * X1 + beta1_true * X2 + np.random.randn(size)

# Specify model using the pymc3 framework
basic_model = Model()
# Reviewing http://docs.pymc.io/Probability_Distributions.html

with basic_model:
    # these are the priors, based on best guesses
    alpha = Normal("alpha", mu=0, sigma=1)  # stock distribution from pymc3
    beta0 = Normal("beta0", mu=12, sigma=1)
    beta1 = Normal("beta1", mu=18, sigma=1)
    # Expectation (conditional mean)
    mu = alpha + beta0*X1 + beta1*X2  # mu here is calculated from priors (it's not updated)
    # Actual sampling distribution of the Y's, which we use for likelihood
    # We sample from the distribution of mu values (which is composed of the a, b0, b1)
    # This step seems to be the accept-reject; we hardcode sigma for some reason.
    Y_obs = Normal("Y_obs", mu=mu, sigma=1, observed=Y)

    trace = sample(1000)

traceplot(trace)

#### Aside to
# https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC2.ipynb
count_data = np.loadtxt("./ignoreland/txtdata.csv")
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.title("Did the user's texting habits change over time?")
plt.xlim(0, n_count_data);

with pm.Model() as model:
    """Define 1 hyperparam, 3 priors"""
    alpha = 1.0 / count_data.mean()
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)

lambda_1.random()

with model:
    """Define data generating model"""
    idx = np.arange(n_count_data)
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)

with model:
    """combine observed data with data-generation model"""
    observation = pm.Poisson("obs", lambda_, observed=count_data)

# We have a set of priors and their functional relationship.  So for any given candidate param values
# for these priors (now posteriors), we can calculate the prior prob of param * new data likelihood on param
# and walk randomly around it, piling up accepted proposed param values (MH takes care of the tendency of this pile
# To resemble the posterior.
# Iterate through (each successive point is drawn from n dist with theta = current point)
# Enter MH to accept or reject certain points based on the updated (prior * likelihood) parameter dist
# for the new/old point.   Prior prob of the point (as a candidate theta) * likelihood of observed data on that point
# (as a candidate theta).  So it's a random walk through thetas to collect point estimates of posterior density at that point

with model:
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000, step=step)

lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
tau_samples = trace['tau']


#histogram of the samples:
ax = plt.subplot(311)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_1$", color="#A60628", density=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables
    $\lambda_1,\;\lambda_2,\;\tau$""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7A68A6", density=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_count_data, alpha=1,
         label=r"posterior of $\tau$",
         color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data)-20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("probability");

