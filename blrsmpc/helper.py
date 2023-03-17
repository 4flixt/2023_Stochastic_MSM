import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

"""
Taken from: 
https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
with minor adaptations.
"""

def plot_cov_as_ellipse(mean_x, mean_y, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    mean_x : float
        Mean of the distribution in x-direction.
    
    mean_y : float
        Mean of the distribution in y-direction.
        
    cov: array-like, shape (2, 2)
        Covariance matrix of the distribution.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def print_percent_done(current, total, metric=None, metric_name = '', bar_len=50, title='Please wait'):
    '''
    Simple progress bar. Optionally prints some metric.

    Args:
        current (int): Current iteration
        total (int): Total number of iterations
        metric (float, optional): Current value of metric
        metric_name (str, optional): Name of metric
        bar_len (int, optional): Length of the progress bar
    '''
    percent_done = (current+1)/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print_msg = f'\t⏳{title}: [{done_str}{togo_str}] {percent_done}% done'

    if metric is not None:
        print_msg += f' - {metric_name}: {metric:.4f}'

    print(print_msg, end='\r')