

"""
#### Notes on Modeling
BG/NBD models probability of death after each purchase. X ~ binomial; p ~ beta
Number of purchases follows Poisson at individual level; N ~ Poisson; lambda ~ gamma
Two variations on this (the purchase model and the survival model, respectively):
BG/BB, which models prob of purchase within each countable period X ~ binomial; p ~ beta
P/NBD models continuous the survival probability, X ~ exponential (vs geom); mu ~ gamma
"""

"""
Notes on business approaches
https://www.seas.upenn.edu/~cse400/CSE400_2013_2014/reports/13_report.pdf
Customer Equity approach, which includes Residual LTV and probability-weighted/intervention-influenced
customer acquisition.  More global model.
Multi-Task learning approach borrows strength across product-category specific models.  
"""
# Following http://www.brucehardie.com/notes/024/reconciling_clv_formulas.pdf
"""
Key distinction between contractual/non, whether first purchase is included,
book at start/end of period.  
Case 1: Contractual => first purchase is included, second occurs contingent on renewal at period k.
So, it's a geometric progression; expected number of renewals.  That's the eCLV of a non-yet acquired customer.
Case 2: Similar, but remove first payment.  That's residual LV for the customer. 
Case 3: Each payment is discounted, because it occurs at end of period. 
"""
# Following http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf
"""
BG-NBD is a computationally more tractable (discrete implementation) of the Gamma-E-NBD model
"""

#### Following paper here: https://srepho.github.io/CLV/CLV
"""
Migration model allows generation of a matrix with transition probabilities
Straightforward to fit Nelson Aalen est with python (lifelines package)
Nelson-Aalen reflects multi-state transitions
Evaluated different models based on couple of traits (hyperparam count, preprocessing, robustness)
PNBD tends to perform better when small cum incidence of purchase, but high repeat rate.
Note on clustering: Hartigan's rule for count of clusters suggesting ratio of within-cluster SS for
k, k+1 clusters.  Threshold is suggested; obv sensitive to scaling, etc.  
"""
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
rossi_dataset = load_rossi()
rossi_dataset.info()
cf = CoxPHFitter(alpha=0.95)
cfit0 = cf.fit(rossi_dataset, duration_col="week", event_col="arrest")
cfit0.print_summary()
axes=cfit0.check_assumptions(rossi_dataset, show_plots=True)

cfit1 = cf.fit(rossi_dataset, duration_col="week", event_col="arrest",
               strata=["wexp"])
cfit1.print_summary()

# cf.fit(formula=)