# Accompanying code repository to our work "Probabilistic multi-step identification with implicit state estimation for stochastic MPC"

## Introduction

Dear reader,

welcome to this repository. You'll find here the code that was created to produce the results for our work "Probabilistic multi-step identification with implicit state estimation for stochastic MPC".
Please don't hesitate to write us a message, if you have any questions. If your questions is of concern to other users, we suggest you use the [discussions tab](https://github.com/4flixt/2023_Stochastic_MSM/discussions).
To better understand the structure of this repository and our code please read the overview below.

## Structure

This repository is structured as follows.

Under [blrsmpc](https://github.com/4flixt/2023_Stochastic_MSM/tree/main/blrsmpc) you'll find the entire source code that was used to create the results. In particular:
- ``system``: The ODE and discretizte formulation of the investigated **building system model** and our simulation tools
- ``sysid``: The required code for probabilistic identification of state-space and multi-step models
- ``SMPC``: Our implementation of an SMPC controller with state-space and multi-step model

Our results can be recreated with the code in [results](https://github.com/4flixt/2023_Stochastic_MSM/tree/main/results). We show two examples:
- A linear building system 
    - In [sid_building_compare_kf_pred](https://github.com/4flixt/2023_Stochastic_MSM/tree/main/results/building_system/sid_building_compare_kf_pred.py), the code to create Figure 1 can be found.
    - In [sid_building_output_fb_for_smpc](https://github.com/4flixt/2023_Stochastic_MSM/tree/main/results/building_system/sid_building_output_fb_for_smpc.py), we identify the multi-step and state-space model for SMPC
    - In [smpc_meta_building](https://github.com/4flixt/2023_Stochastic_MSM/tree/main/results/building_system/smpc_meta_building.py), we perform SMPC with the identified models and the results in **Figure 2** and **Table 1** are created.
- A nonlinear CSTR system
    - In [sid_cstr](https://github.com/4flixt/2023_Stochastic_MSM/tree/main/results/building_system/sid_cstr.py), we identify the multi-step and state-space model for the SMPC. The results shown in **Figure 3** are created. 
    - In [smpc_cstr](https://github.com/4flixt/2023_Stochastic_MSM/tree/main/results/building_system/smpc_cstr.py), the results shown in **Figure 4** and **Table 3** are created.
  
We strive to make our results as accesible as possible and are happy to get feedback.

