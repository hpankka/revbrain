import numpy as np
import matplotlib.pyplot as plt


def plot_mae(mae, group_delay, ms_per_sample):

    c = '#D4575E'

    font_s = 16
    plt.rcParams.update({'font.size': font_s})

    fig, ax = plt.subplots()

    time_axis = np.arange(1, len(mae)+1)
    ax.plot(time_axis, mae, linewidth=2.5, color=c)

    ymax = max(mae)+1
    ax.fill_between([0, group_delay], [0, 0], [ymax, ymax], color='#FAEBD7', alpha=0.5, zorder=2.5)
    ax.vlines(x=group_delay, ymin=0, ymax=ymax, color='grey', zorder=3)

    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_position(("data", 0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim([0, len(mae)+3])
    ax.set_xlabel('Time (ms)')
    x_ticks = list(np.arange(0, len(mae), 10))
    x_labels = [round(t*ms_per_sample - group_delay*ms_per_sample) for t in x_ticks]
    x_labels[::2] = [""]*(len(x_labels[::2]))
    ax.set_xticks(x_ticks, x_labels)
    ax.set_ylim([0, ymax])
    ax.set_ylabel('MAE (\u03BCV)')
    y_ticks = np.arange(0.5, ymax, 0.5)
    y_labels = list(y_ticks.astype(int))
    y_labels[0::2] = [""] * len(y_labels[0::2])
    ax.set_yticks(y_ticks, y_labels)
    ax.tick_params(width=1.5)
    
    ax.set_title('MAE as a function of time')

    fig.tight_layout()

    return fig

def plot_recall_precision(precision, recall, ths):

    c_precision = '#445D91'
    c_recall = '#4E9E6E'

    font_s = 16
    plt.rcParams.update({'font.size': font_s})

    fig, ax = plt.subplots()

    ax.plot(ths, precision, label='precision', linewidth=2.5, color=c_precision)
    ax.plot(ths, recall, label='recall', linewidth=2.5, color=c_recall)

    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.5)
    ax.set_xlim([49, 100])
    ax.set_xlabel('Threshold %')
    ax.set_ylim([0, 100])
    ax.set_ylabel('%', rotation=0, labelpad=12)

    ax.legend(frameon=False)

    fig.tight_layout()

    return fig
