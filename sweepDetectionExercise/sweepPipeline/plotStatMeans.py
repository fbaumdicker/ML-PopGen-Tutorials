import sys, os, gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

hardStatFileName, softStatFileName, plotFileName = sys.argv[1:]

def readStatMeansFromFile(statFileName, numSubWins=11):
    first=True
    with open(statFileName, "rt") as statFile:
        for line in statFile:
            if first:
                header = line.strip().split()
                statSums = [0]*len(header)
                statTots = [0]*len(header)
                first = False
            else:
                statvals = [float(x) for x in line.strip().split()]
                for i in range(len(statvals)):
                    statSums[i] += statvals[i]
                    statTots[i] += 1

    statMeans = {}
    for i in range(len(header)):
        statName = header[i].split("_win")[0]
        if not statName in statMeans:
            statMeans[statName] = [0]*numSubWins

    for i in range(len(header)):
        statName, win = header[i].split("_win")
        win = int(win)
        statMeans[statName][win] = statSums[i]/statTots[i]
    return statMeans

def plotBigStats(stats, statNames, titles, colors, markers, plotFileName):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for sweepTypeIndex in range(2):
        for i in range(len(statNames)):
            print(sweepTypeIndex, statNames[i], stats[sweepTypeIndex][statNames[i]])
            ax[sweepTypeIndex].plot(list(range(-5,6)), stats[sweepTypeIndex][statNames[i]], color=colors[i], lw=1, marker=markers[i], label=statNames[i])
            plt.setp(ax[sweepTypeIndex].get_xticklabels(), fontsize=14)
            plt.setp(ax[sweepTypeIndex].get_yticklabels(), fontsize=14)
            ax[sweepTypeIndex].set_xlabel("Distance from sweep", fontsize=14)
            ax[sweepTypeIndex].set_ylabel("Relative value of statistic", fontsize=14)
            ax[sweepTypeIndex].set_xticks(list(range(-5,6)))
            ax[sweepTypeIndex].set_xticklabels([""]*5 + ["0"] + [""]*5)
            ax[sweepTypeIndex].legend()
            ax[sweepTypeIndex].set_ylim((0, 0.15))
            ax[sweepTypeIndex].set_title(titles[sweepTypeIndex])
    fig.tight_layout()
    fig.savefig(plotFileName)

colors = ['black','red','blue','violet','orange','cyan','gray','brown']
markers = ['o', 'v', '^', 'x', 's', '+', 'D', '']
statsToPlot = 'pi tajD fayWuH maxFDA HapCount ZnS Omega'.split()

hardStatMeans = readStatMeansFromFile(hardStatFileName)
softStatMeans = readStatMeansFromFile(softStatFileName)
plotBigStats([hardStatMeans, softStatMeans], statsToPlot, ["Hard Sweep", "Soft Sweep"], colors, markers, plotFileName)
