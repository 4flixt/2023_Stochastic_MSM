# Accompanying code repository to our work "Stochastic MPC with multi-step predictors and covariance estimation via Bayesian linear regression"

## Introduction

Dear reader,

welcome to this repository. You'll find here the code that was created to produce the results for our work "Stochastic MPC with multi-step predictors and covariance estimation via Bayesian linear regression".
Please don't hesitate to write us a message, if you have any questions. If your questions is of concern to other users, we suggest you use the [discussions tab](https://github.com/4flixt/2023_Stochastic_MSM/discussions).
To better understand the structure of this repository and our code please read the overview below.

## Structure

This repository is structured as follows.

Under [blrsmpc](https://github.com/4flixt/2023_Stochastic_MSM/tree/main/blrsmpc) you'll find the entire source code that was used to create the results. In particular:
- ``system``: The ODE and discretizte formulation of the investigated **building system model** and our simulation tools
- ``sysid``: The required code for Bayesian linear regression of state-space and multi-step models
- ``SMPC``: Our implementation of an SMPC controller with state-space and multi-step model

Our results can be recreated with the code in [results](https://github.com/4flixt/2023_Stochastic_MSM/tree/main/results)
- In [sid_meta_building.ipynb](https://github.com/4flixt/2023_Stochastic_MSM/blob/main/results/sid_meta_building.ipynb) we identify state-space and multi-step models for the investigated building system for different variants of process and measurement noise. **Figure 1** in our paper was obtained from this code.
- In [smpc_meta_building](https://github.com/4flixt/2023_Stochastic_MSM/blob/main/results/smpc_meta_building.py) we investigate the SMPC controller with state-space and multi-step model. 
  - Open-loop predictions with samples from the true system (**Figure 2** in the paper)
  - Detailed view of the state-space for T1 and T2 to highlight the importance of the full covariance information (**Figure 3** in the paper)
  - CLosed-loop results over 50h for all investigated controller variants (**Table 1** in the paper
  
We strive to make our results as accesible as possible and are happy to get feedback.

