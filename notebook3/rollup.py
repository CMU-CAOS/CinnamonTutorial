import sys
import re

def get_exec_time(file):
    with open(file,'r') as f:
        Lines = f.readlines()
    line = Lines[-1]
    se = re.search(r'Simulation is complete, simulated time: (\d+(.\d+)?) (us|ms|s)', line)
    if se:
        time = float(se.group(1))
        unit = se.group(3)
        if unit == 'us':
            time = time/1000
        elif unit == 's':
            time = time*1000
    return time


def get_time(filename):

    file = f"{filename}"
    try:
        time = get_exec_time(file)
    except FileNotFoundError as ex:
        print(ex,file=sys.stderr)
    return time

def get_times(expt_dir,expts,bandwidths):


    ExecTimes = {}
    for e in expts:
        times = []
        for bw in bandwidths:
            times.append(get_time(f"{expt_dir}/{e}/{bw}bw.log"))
        ExecTimes[e] = times

    return ExecTimes


import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.rcParams.update({'font.size': 8, 'font.family':'serif'})
# matplotlib.rcParams.update({'figure.figsize': (4.2,2.3)})

def plot_speedups(expt_dir,experiments,bandwidths):

    ExecTimes = get_times(expt_dir,experiments,bandwidths)

    max = 0
    Speedups = {}
    for k,v in ExecTimes.items():
        lmax = np.max(v)
        max = np.max((max,lmax))

    maxSpeedup = 0
    Speedups = {}
    for k,v in ExecTimes.items():
        Speedups[k] = [max/t for t in v]
        maxSpeedup = np.max((maxSpeedup,np.max(Speedups[k])))

    filled_marker_style = dict(marker="o", linestyle="-", markersize=6,
                            markeredgecolor='k')
    fig, ax = plt.subplots()
    for e in experiments:
        ax.plot(bandwidths,Speedups[e],label=f'{e}',fillstyle='full',**filled_marker_style)


    plt.yscale('log',base=2)
    ax.legend()
    ax.grid(axis='y',linestyle=':')
    ticks = [int(2**i) for i in range(int(np.log2(maxSpeedup)) + 1)]
    ax.set_yticks(ticks,ticks)

    ax.set_ylabel('Normalized Speedups')
    ax.set_xlabel('Bandwidths')

    plt.tight_layout()
    plt.show()
