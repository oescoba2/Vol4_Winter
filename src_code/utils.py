from numpy.typing import ArrayLike, NDArray
from typing import Callable
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def drug_concentration(time:float|ArrayLike, doses:ArrayLike|list, decay_rate:float, duration:int=6, 
                       dose_interval:int=21) -> float|ArrayLike:
    """Compute the function of a drug quantity for a given time using the model chemo
    model proposed by Ophir Nave in
    https://pmc.ncbi.nlm.nih.gov/articles/PMC9065634/ .

    Parameters:
        - time (float or ArrayLike)
        - doses (float) : an array or list containing the the quantities administered 
                          for the duration of the chemo treatment.
        - decay_rate (float) : the terminal half life of the drug measured in (1/days).
        - duration (int) : the duration of chemo treatment which is the total number of 
                           doses the patient will receive. Defaulted to 6.
        - dose_interval (int) : the number of days between administered doses. Defaulted
                                to 21.
    
    Returns:
        - f_t (float or ArrayLike) : the generated concentration of the drug over time.
    """

    time = np.asanyarray(time)
    f_t = np.zeros_like(time, dtype=float)

    for i in range(duration):
        delay = i * dose_interval
        step = np.heaviside(time - delay, 1.0)            # Unit step
        decay = np.exp(-np.log(2)/decay_rate * (time - delay))
        f_t += doses[i] * step * decay

    return f_t

def plot_drugs(time:ArrayLike, drug_vals:dict[str, ArrayLike]|ArrayLike, colors:list[str]=None,
               xlabel='Time (days)', ylabel:str='Concentration ($\\frac{mg}{mL}$)',
               figtitle:str='Chemotherapy Concentration.') -> None:
    """Plot a generated concentration of one drug or several of them over time. 

    Parameters:
        - time (ArrayLike) : the time values at which the drug is measured.
        - drug_vals (dict or ArrayLike) : either an ArrayLike or dictionary containing the 
                                          values of the drug or drugs to plot over time.
                                          If drug_vals is an np.array, it is assumed that only
                                          one drug will be plotted. If drug_vals is a dictionary,
                                          then several drugs will be plotted. In this last case,
                                          the dictionary must have as keys strings denoting the 
                                          name of the drugs and the values are arrays with the 
                                          concentrations of the drugs over time.
        - colors (list) : a string or a list of strings denoting the color or colors to use to 
                          plot the given drug or drugs. Defaulted to None.
        - xlabel (str) : the label to give the x-axis of the generated plot. Defaulted to 'Time (days)'.
        - ylabel (str) : the label to give the y-axis of the generated plot. Defaulte to 
                         r'Concentration ($\frac{\text{mg}}{\text{mL}}$)'.
        - figtitle (str) : the title to give the generated plot. Defaulted to 'Chemotherapy Concentration'.
        
    Returns:
        - None
    """

    if isinstance(drug_vals, dict):
        multiple_drugs = True
    elif isinstance(drug_vals, np.array):
        multiple_drugs = False

    ax = plt.subplot(111)

    if multiple_drugs:

        drugs = list(drug_vals.keys())
        vals = list(drug_vals.values())

        if colors is None:
            for drug, val in zip(drugs, vals):
                ax.plot(time, val, label=drug)
        else:
            for drug, val, color, in zip(drugs, vals, colors):
                ax.plot(time, val, label=drug, color=color)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(figtitle)
        ax.legend()

    else:

        if colors is None:
            ax.plot(time, drug_vals, color=colors)
        else:
            ax.plot(time, drug_vals)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(figtitle)

    plt.show()

def make_ani(time:ArrayLike, normalized_states:NDArray, tf:float|int, useGrid:bool, title:str,
             xlabel:str) -> None:
    """Function to animate the normalized state dynamics over time.
    
    Parameters:
        - time (ArrayLike) : the time of the system evolutions.
        - states (NDArray) : the system states evolved through time.
        - tf (float|int) : the final time of the system. Used to set up the x-limits
        - useGrid (bool) : whether to use a grid or not. 
        - title (str) : a str to use to plot the title of the figure.
        - xlabel (str) : the labeling to use for the x-axis.
         
    Returns:
        - None:
    """

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    colors = ['red', 'blue', 'orange', 'green']
    labels = ['IDC', 'Normal', 'CD8$^+$', 'NK']
    lines = [ax.plot([], [], color=color, label=label)[0] for color, label in zip(colors, labels)]

    ax.set_xlim(-0.001, tf+0.01)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('% of Population')

    if useGrid:
        ax.grid(True, alpha=0.3)

    def update(idx) -> plt.axes:
        """Function to update the fig object.
        
        Parameters:
            - idx (int) : the index to plot on the lines
        Returns:
            - (plt.axes) : the updated line objects
        """

        for line, state in zip(lines, normalized_states):
            line.set_data(time[:idx+1], state[:idx+1])
            
        return lines

    ax.legend()
    ax.set_title(title)
    animate = animation.FuncAnimation(fig=fig, func=update, frames=range(len(time)), interval=50)
    animate.save('./imgs/system_dynamics.mp4')
    plt.close()


def plot_state_dynamics(state_sol:Callable, t0:int=0, t1:int=126, tf:int=156, linecolor:str='blue', 
                        xscale:str='linear', yscales:list[str]=['linear','linear','linear','linear'], 
                        xlabel:str='Time (Days)', 
                        ylabels:list[str]=['Number of IDC Cells','Number of Normal Cells', 'Number of CD$8^{+}$Cells', 'Number of NK Cells'],
                        title:str='State Dynamics', normalize:bool=False, plot_end_chemo:bool=False,
                        use_grid:bool=False, animate:bool=False, save_fig:bool=False, 
                        fig_name:str='state_dynamics.pdf') -> None:
    """Plot the state dynamics in 2x2 grid. Plots are arranged in the following
    order (for unnormalized):

    Tumor Normal
    CD    NK
    
    Parameters:
        - state_sol (Callable) : a callable object from scipy.integrate that will
                                 allow the computation of the states.
        - t0 (int) : the starting time for the modeling problem. Defaulted to 0.
                     This corresponds to the lowertime  bound of the cost functional.
        - t1 (int) : the upper time bound of the cost functional. Defauled to 126.
                     This is used to plot a vertical line to indicate this time.
        - tf (int) : the final time for the modeling problem. Defaulted to 156. This
                     can be bigger than the t1. This is used to continue evolving
                     the system past t1.
        - linecolor (str) : the line color to use on the plot. Defaulted to 'blue'.
        - xscale (str) : the scale to use on the x-axis for plotting. Defaulted to
                         'linear'.
        - yscales (list) : a list of the scales to use on the y-axis for plotting for
                           each generated subplot. Defaulted to
                           ['linear', 'linear', 'linear', 'linear']. Note that the 
                           first value corresponds to the first state component and
                           successively.       
        - xlabel (str) : the label to use on the x-axis. Defaulted to 'Time (Days)'.         
        - ylabels (list) : a list of string containing the labels to use on the y-axis
                           of each plot. Note that the first value corresponds to the
                           first state component and successively. Defaulted to
                           ['Number of IDC Cells','Number of Normal Cells', 
                           'Number of CD$8^{+}$Cells', 'Number of NK Cells'].
        - title (str) : the title to give the overall produced image. Defaulted to 
                        'State Dynamics'.
        - normalize (bool) : whether to normalize the states or not. Defauled to False.
                             If True, only one plot is produced.
        - plot_end_chemo (bool) : whether to plot a vertical line indicating the end
                                  of AC treatment. Defaulted to False.
        - use_grid (bool) : whether to use a grid or not. Defaulted to False.
        - save_fig (bool) : a boolean specifying whether or not to save the generated
                            figure. Defaulted to False.
        - fig_name (bool) : the name to give the generated figure. Defaulted to 
                            './imgs/state_dynamics.pdf'

    Returns:
        - None
    """
    

    if normalize:
        fig, ax = plt.subplots(figsize=(12, 6))
        time = np.linspace(t0, tf, num=400)
        states = state_sol.sol(time)

        total_pop = states.sum(axis=0)
        states_norm = states / total_pop
        ax.plot(time, states_norm[0], color='red', label='IDC')
        ax.plot(time, states_norm[1], color='blue', label='Normal')
        ax.plot(time, states_norm[2], color='orange', label='CD8$^+$')
        ax.plot(time, states_norm[3], color='green', label='NK')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('% of Population')
        ax.set_title(title)
        ax.legend()

        if use_grid:
            ax.grid(True, alpha=0.3)

        if plot_end_chemo:
            ax.axvline(x=t1, ymin=0, ymax=0, linestyle='dashed', color='purple', label='End of Chemo')

        if save_fig:
            plt.savefig(fig_name, format='pdf')


        if animate:
            make_ani(time=time, normalized_states=states_norm, tf=tf, useGrid=use_grid, title=title, xlabel=xlabel)

        plt.show()

    else:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
        time = np.linspace(t0, tf, num=400)
        states = state_sol.sol(time)

        ax[0, 0].plot(time, states[0], color=linecolor)
        ax[0, 0].set_xlabel(xlabel)
        ax[0, 0].set_ylabel(ylabels[0])
        ax[0, 0].set_xscale(xscale)
        ax[0, 0].set_yscale(yscales[0])

        ax[0, 1].plot(time, states[1], color=linecolor)
        ax[0, 1].set_xlabel(xlabel)
        ax[0, 1].set_ylabel(ylabels[1])
        ax[0, 1].set_xscale(xscale)
        ax[0, 1].set_yscale(yscales[1])

        ax[1, 0].plot(time, states[2], color=linecolor)
        ax[1, 0].set_xlabel(xlabel)
        ax[1, 0].set_ylabel(ylabels[2])
        ax[1, 0].set_xscale(xscale)
        ax[1, 0].set_yscale(yscales[2])  

        ax[1, 1].plot(time, states[3], color=linecolor)
        ax[1, 1].set_xlabel(xlabel)
        ax[1, 1].set_ylabel(ylabels[3])
        ax[1, 1].set_xscale(xscale)
        ax[1, 1].set_yscale(yscales[3])   

        if plot_end_chemo:
            ax[0, 0].axvline(x=t1, ymin=0, ymax=states[0].max(), linestyle='dashed', color='red', label='End of Chemo')
            ax[0, 1].axvline(x=t1, ymin=0, ymax=states[1].max(), linestyle='dashed', color='red', label='End of Chemo')
            ax[1, 0].axvline(x=t1, ymin=0, ymax=states[2].max(), linestyle='dashed', color='red', label='End of Chemo')
            ax[1, 1].axvline(x=t1, ymin=0, ymax=states[3].max(), linestyle='dashed', color='red', label='End of Chemo')

        plt.tight_layout()
        plt.suptitle(title, va='top')

        if save_fig:
            plt.savefig(fig_name, format='pdf')

        plt.show()