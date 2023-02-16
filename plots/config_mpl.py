import matplotlib as mpl

fontsize = 10
mpl.rcParams.update({
    'lines.linewidth':1,
    'font.size': fontsize,
    'axes.titlesize'  : fontsize,
    'axes.labelsize'  : fontsize,
    'xtick.labelsize' : fontsize, 
    'ytick.labelsize' : fontsize,
    'font.serif': [],
    'axes.titlepad':  10,
    'axes.labelsize': 'medium',
    'figure.figsize': (7.15, 4),
    'axes.grid': False,
    'lines.markersize': 5,
    'text.usetex': True,
    'pgf.rcfonts': False,    # don't setup fonts from rc parameters
    'axes.unicode_minus': False,
    "text.usetex": True,     # use inline math for ticks
    "pgf.texsystem" : "xelatex",
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
})


# Load notation from file. The same notation is used in the paper.
with open('../Plots/notation.tex', 'r') as f:
    tex_preamble = f.readlines()

tex_preamble = ''.join(tex_preamble)

mpl.rcParams.update({
    'text.latex.preamble': tex_preamble,
    'pgf.preamble': tex_preamble,
})

color = mpl.rcParams['axes.prop_cycle'].by_key()['color']
boxprops = dict(boxstyle='round', facecolor='white', alpha=0.5, pad = .4)