
import matplotlib
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from matplotlib import ticker

################################################
# Figures configuration
################################################


def use_default_style() -> None:
    mpl.rcdefaults()  # Reset to the default Matplotlib settings


def use_latex_style(bool=True) -> None:
    if bool:
        # Figures configuration
        mpl.use("pgf")
        mpl.rcParams.update({
            "pgf.texsystem": "pdflatex",  # Change this if using xelatex or lualatex
            "font.family": "serif",  # Use a more commonly available font family
            "mathtext.fontset": "stix",  # Use STIX for math text
            "font.size": 17,  # General font size
            "axes.titlesize": 17,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,  # Specific font size for x-tick labels
            "ytick.labelsize": 15,  # Specific font size for y-tick labels
            "legend.fontsize": 15,
            "figure.titlesize": 17,
            "text.usetex": True,  # Use LaTeX to write all text
            "pgf.rcfonts": False,  # Don't setup fonts from rc parameters
            # Use a raw string for the preamble
            "pgf.preamble": r"\usepackage{amsmath}",
        })


def use_latex_style_2(bool=True) -> None:
    if bool:
        # Reset defalut values
        plt.rcdefaults()
        mpl.rc_file_defaults()
        # Figures configuration
        matplotlib.pyplot.rc('font', **{'family': 'sans-serif',
                             'sans-serif': ['Computer Modern Sans serif']})
        mpl.rcParams.update(
            {'font.size': 7, 'font.family': 'STIXGeneral', 'text.usetex': False})

################################################


def EFI_evolution(Ed_removed, detFIM, n_sensors, n_modes, title):
    """
    Function Duties:
        Plots the evolution of the EFI algorithm's parameters:
            the minimum value of Ed for each iteration
            the determinant of the FIM matrix for each iteration.
    Input Parameters:
        Ed_removed: list of floats
            the minimum value of Ed for each iteration.
        detFIM: list of floats
            the determinant of the FIM matrix for each iteration.
        n_sensors: int
            the number of sensors.
        n_modes: int
            the number of modes.
        title: str
            identifiable title (e.g. SG or Accelerometers).
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax[0, 0].plot(Ed_removed)
    ax[0, 0].set_xlabel('Iteration')
    ax[0, 0].set_ylabel("Removed sensor's Ed")
    ax[0, 0].set_title('Minimum value of Ed for each iteration')
    ax[0, 0].grid()
    ax[0, 1].plot(Ed_removed)
    ax[0, 1].set_xlim([len(Ed_removed)-10, len(Ed_removed)])
    ax[0, 1].set_xlabel('Iteration')
    ax[0, 1].set_ylabel("Removed sensor's Ed")
    ax[0, 1].set_title('Minimum value of Ed for each iteration - zoomed')
    ax[0, 1].grid()
    ax[1, 0].plot(detFIM)
    ax[1, 0].set_xlabel('Iteration')
    ax[1, 0].set_ylabel('det(FIM)')
    ax[1, 0].set_title('Determinant of FIM matrix for each iteration')
    ax[1, 0].grid()
    ax[1, 1].plot(detFIM)
    ax[1, 1].set_xlim([len(detFIM)-10, len(detFIM)])
    ax[1, 1].set_xlabel('Iteration')
    ax[1, 1].set_ylabel('det(FIM)')
    ax[1, 1].set_title('Determinant of FIM matrix for each iteration - zoomed')
    ax[1, 1].grid()
    fig.suptitle(
        f"EFI Algorithm's parameters evolution: {n_sensors} {title}; {n_modes} modes")
    fig.tight_layout()

    return fig, ax


def EFI_evolution_simplified(Ed_removed, detFIM, n_sensors, n_modes, title):
    """
    Function Duties:
        Analogous to EFI_evolution, but with only 2 subplots.
    """
    use_latex_style(True)
    total_number_sensors = np.arange(len(detFIM) + n_sensors - 1, n_sensors - 1, -1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(total_number_sensors, Ed_removed)
    ax[0].set_xlabel('Number of sensors')
    ax[0].set_ylabel(r"Smallest $E_D$ value")
    ax[0].set_title(r"$E_D$ minimum value - evolution for each iteration")
    ax[0].grid()
    ax[0].set_xlim([total_number_sensors[0], total_number_sensors[-1]])
    ax[1].plot(total_number_sensors, Ed_removed)
    ax[1].set_xlim([total_number_sensors[-10], total_number_sensors[-1]])
    ax[1].set_xlabel('Number of sensors')
    ax[1].set_ylabel("Smallest $E_D$ value")
    ax[1].set_title(r"$E_D$ minimum value - evolution for each iteration (zoomed)")
    ax[1].grid()
    # fig.suptitle(
    #     f"EFI Ed evolution: OSP of {n_sensors} {title}; {n_modes} target modes")
    fig.tight_layout()

    return fig, ax


def plot_MAC_1(MAC, language='Spanish'):

    n_modes = np.shape(MAC)[0]

    if language == 'Spanish':
        label = 'Modo'
    else:
        label = 'Mode'

    col = list()
    for kk in range(n_modes):
        col.append(label + ' ' + str(kk+1))

    MAC = pd.DataFrame(MAC, columns=col, index=col)
    fig, ax = plt.subplots()
    sns.heatmap(MAC, cmap="jet", ax=ax, annot=True, fmt='.3f')
    fig.tight_layout()

    return (fig, ax)


def plot_MAC_2(MAC, language='Spanish'):

    if language == 'Spanish':
        title = 'Matriz MAC'
    else:
        title = 'MAC Matrix'

    n_modes = np.shape(MAC)[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    top = MAC.reshape((np.size(MAC), 1))
    bottom = np.zeros_like(top)
    max_height = np.max(top)   # get range of colorbars so we can normalize
    min_height = np.min(top)

    x = np.array([np.arange(1, n_modes+1), ] *
                 n_modes).reshape((np.size(top), 1))-0.5
    y = np.array([np.arange(1, n_modes+1), ] *
                 n_modes).transpose().reshape((np.size(top), 1))-0.5

    width = depth = 1
    # cmap = plt.cm.get_cmap('jet')  # will be deprecated, use better matplotlib
    # Get desired colormap - you can change this!
    cmap = matplotlib.colormaps.get_cmap('jet')

    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in top[:, 0]]
    ax.bar3d(x[:, 0], y[:, 0], bottom[0, :], width,
             depth, top[:, 0], shade=True, color=rgba)
    ax.set_title(title)

    return (fig, ax)


def plot_ConvDiag_formatted(Targt, acc_rate):
    """
    Function duties:
        Plot_1: number of iterations vs expectation(Targt)
        Plot_2: number of iterations vs acceptance rate
    Input:
        Targt: list containing likelihood related to each set of parameters
        acc_rate: number of accepted / total number of iterations
    """
    use_latex_style(True)
    # A) Expectation computation
    Expectation = np.cumsum(Targt)/np.arange(1, len(Targt)+1)
    # Expectation=cumsum(cumsum(Targt)/arange(1,len(Targt)+1)/arange(1,len(Targt)+1)) #performs another level of average for in case of large variability in target

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=100)

    axes[0].plot(Expectation, color='black', lw=0.9)
    axes[0].set_xlabel('Number of iterations')
    axes[0].set_title((r'Log-likelihood expectation'))
    axes[0].grid(True, ls=":")

    axes[1].plot(acc_rate, color='black', lw=0.9)
    axes[1].set_xlabel('Number of iterations')
    axes[1].set_title(r'Acceptance rate')
    # axes[1].set_ylim([0,1.1])
    axes[1].grid(True, ls=":")
    fig.tight_layout()

    return fig, axes


def plot_ConvDiag(Targt, acc_rate):
    """
    Function duties:
        Plot_1: number of iterations vs expectation(Targt)
        Plot_2: number of iterations vs acceptance rate
    Input:
        Targt: list containing likelihood related to each set of parameters
        acc_rate: number of accepted / total number of iterations
    """
    use_latex_style_2(True)
    # A) Expectation computation
    Expectation = np.cumsum(Targt)/np.arange(1, len(Targt)+1)
    # Expectation=cumsum(cumsum(Targt)/arange(1,len(Targt)+1)/arange(1,len(Targt)+1)) #performs another level of average for in case of large variability in target

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), dpi=100)

    axes[0].plot(Expectation, color='black', lw=0.9)
    axes[0].set_title(r'Target expectation', fontsize=11)
    axes[0].grid(True, ls=":")

    axes[1].plot(acc_rate, color='black', lw=0.9)
    axes[1].set_title(r'Acceptance rate', fontsize=11)
    # axes[1].set_ylim([0,1.1])
    axes[1].grid(True, ls=":")
    fig.tight_layout()

    return fig, axes


def plot_Parameters_Posterior(PostParam):
    """
    Function Duties:
        Plots the posterior distribution of the parameters
    Input:
        PostParam: np array of dimension m, n:
            m: number of iterations
            n: number of parameters
    IMPORTANT:
        Ensure that PostParam is chosen after the burn-in period
    """
    use_latex_style_2(True)
    numvars = np.shape(PostParam)[1]
    fig, axes = plt.subplots(nrows=1, ncols=numvars, figsize=(
        13, 3.5), sharex='col')  # ,sharex='col', sharey='row')
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i in range(numvars):
        data = PostParam[:, i]
        kde_pdf = sp.stats.gaussian_kde(data, bw_method='scott')

        rnge = np.ptp(data)  # (equiv. to np.max(data)-np.min(data))
        MIN, MAX = min(data)-rnge/10, max(data)+rnge/10
        n = 2**6  # number of discritization points
        Rnge = MAX-MIN
        xmesh = MIN+np.linspace(0, Rnge, n)

        axes[i].hist(data, bins='auto', density=True, stacked=True,
                     edgecolor='0.65', facecolor='0.65')  # density=True,
        axes[i].plot(xmesh, kde_pdf(xmesh), linewidth=0.8,
                     color='black', alpha=0.7)

        # Set scientific notation, labels, etc.
        axes[i].set_xlabel(r'$\theta_{'+str(i+1)+'}$')
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axes[i].xaxis.set_major_formatter(formatter)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axes[i].yaxis.set_major_formatter(formatter)
        axes[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        axes[i].yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        axes[i].grid(True, ls=":")
        if i == 0:
            axes[0].set_ylabel(r'$p(\theta_i|\mathcal{D},\mathcal{M})$')

    fig.tight_layout()

    return fig, axes


def plot_Parameters_Posterior_formatted(PostParam):
    """
    Function Duties:
        Plots the posterior distribution of the parameters
    Input:
        PostParam: np array of dimension m, n:
            m: number of iterations
            n: number of parameters
    IMPORTANT:
        Ensure that PostParam is chosen after the burn-in period
    """
    use_latex_style(True)
    numvars = np.shape(PostParam)[1]
    fig, axes = plt.subplots(nrows=1, ncols=numvars, figsize=(
        13, 3.5), sharex='col')  # ,sharex='col', sharey='row')
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i in range(numvars):
        data = PostParam[:, i]
        kde_pdf = sp.stats.gaussian_kde(data, bw_method='scott')

        rnge = np.ptp(data)  # (equiv. to np.max(data)-np.min(data))
        MIN, MAX = min(data)-rnge/10, max(data)+rnge/10
        n = 2**6  # number of discritization points
        Rnge = MAX-MIN
        xmesh = MIN+np.linspace(0, Rnge, n)

        axes[i].hist(data, bins='auto', density=True, stacked=True,
                     edgecolor='0.65', facecolor='0.65')  # density=True,
        axes[i].plot(xmesh, kde_pdf(xmesh), linewidth=0.8,
                     color='black', alpha=0.7)

        # Set scientific notation, labels, etc.
        axes[i].set_xlabel(r'$\theta_{'+str(i+1)+'}$')
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axes[i].xaxis.set_major_formatter(formatter)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axes[i].yaxis.set_major_formatter(formatter)
        axes[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        axes[i].yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        axes[i].grid(True, ls=":")
        # axes[i].xaxis.get_offset_text().set_x(1.1)  # Adjust the x-position of the exponent
        # axes[i].xaxis.get_offset_text().set_fontsize(8)  # Set a smaller font size for the exponent
        if i == 0:
            axes[0].set_ylabel(r'$p(\theta_i|\mathcal{D},f)$')

    fig.tight_layout()

    return fig, axes


def plot_Joint_PDF(PostParam, step=100):
    """
    Function Duties:
        Plots the joint PDF of the parameters
    Input:
        PostParam: np array of dimension m, n:
            m: number of iterations
            n: number of parameters
        step: one data over 'step' is considered to avoid correlation
    Returns:
        plot_grid: seaborn PairGrid object
    """
    # Ignore warnings related to seaborn - pandas interaction
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="seaborn._oldcore")
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn._oldcore")
    
    # Select only some of the data to avoid correlation and arrange into a df
    PostParamSimp = PostParam[::step]
    df = pd.DataFrame(PostParamSimp, columns=[r'$\theta_{'+str(i+1)+'}$' for i in range(np.shape(PostParamSimp)[1])])

    # Plot the joint PDF
    style = {'font.family': 'sans-serif', 'grid.linestyle': ':', 'axes.grid': True}
    sns.set_style('white', style)  # available are darkgrid, whitegrid, dark, white, and ticks
    sns.axes_style()  # to know the defaul style
    sns.set_context("paper")
    plot_grid = sns.PairGrid(df, diag_sharey=True)

    plot_grid.map_diag(plt.hist)
    plot_grid.map_lower(sns.kdeplot, cmap="Blues_d", n_levels=4)

    base_color = "#1f77b4"  # blue
    edge_color = modify_color(base_color, factor=-0.4)  # lighten the color (factor < 0)
    plot_grid.map_upper(plt.scatter, color=base_color, edgecolor=edge_color)

    # Set scientific notation, etc.
    for ax in plot_grid.axes.flat:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(formatter)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    plot_grid.figure.tight_layout()

    return plot_grid


def modify_color(color, factor=0.2):
    """
    Function Duties:
        Darkens a given color by reducing its RGB values.
    Parameters:
        color: str or tuple, color to darken (e.g., "#1f77b4" or (0.1, 0.2, 0.3))
        factor: float, the amount to darken the color
             0 = no change
             among 0 and 1 = darken
             among 0 and -1 = lighten
    Returns:
        modified color in hex format
    """
    # Convert the color to an RGB tuple if it's in hex format
    rgb = matplotlib.colors.hex2color(color) if isinstance(color, str) else color
    # Darken the color by reducing each RGB component
    darkened_rgb = tuple(max(0, c * (1 - factor)) for c in rgb)
    # Convert back to hex format
    return matplotlib.colors.to_hex(darkened_rgb)
