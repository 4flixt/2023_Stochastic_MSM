# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os

sys.path.append(os.path.join('..'))
import helper

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# %%

T1_lb = 18

T1 = np.arange(17, 21, 1)
T1_feasible = np.arange(T1_lb, 21, 1)

# T2 -T1 >= 1
T2_ub = T1 + 1
T2_ub_feasible = T1_feasible + 1

# T2 - T1 <= 3
T2_lb = T1 + 4
T2_lb_feasible = T1_feasible + 4
# %%

fig, ax = plt.subplots(figsize=(4,5.5))

ax.axis('equal')

ax.plot(T1, T2_ub, color='k', linestyle='--', label='constraints')
ax.plot(T1, T2_lb, color='k', linestyle='--')
ax.axvline(T1_lb, color='k', linestyle='--')
ax.fill_between(T1_feasible, T2_ub_feasible, T2_lb_feasible, color='k', alpha=0.1, label='feasible set (nominal)')


cov1 = np.array([[1, 0.95], [1, .95]])
cov2 = np.diag(np.diag(cov1))

mean_1 = (18.9, 20.05)
mean_2 = (18.9, 21.2)

helper.plot_cov_as_ellipse(*mean_2, cov2, n_std=1, ax=ax, color=colors[1], alpha=0.5, label='est. variance')
helper.plot_cov_as_ellipse(*mean_1, cov1, n_std=1, ax=ax, color=colors[0], alpha=0.5, label='est. covariance')

case_1_offset = T2_ub + mean_1[1] - mean_1[0] - 1
case_2_offset = T2_ub + mean_2[1] - mean_2[0] - 1

ax.plot(T1, case_1_offset, color=colors[0], linestyle='--', label='constraint offset (covariance est.)')
ax.plot(T1, case_2_offset, color=colors[1], linestyle='--', label='constraint offset (variance est.)')

ax.set_xlabel('Temp. room 1 [°C]')
ax.set_ylabel('Temp. room 2 [°C]')

ax.text(18.5, 18, "$t_2 - t_1 \geq 1$\n $t_2-t_1 \leq 4$", fontsize=12)

ax.legend()


# %%
