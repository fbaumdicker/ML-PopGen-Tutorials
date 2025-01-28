import os, sys
import numpy as np
from collections import Counter
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

classifierPath, testDir, predFile, figFile = sys.argv[1:]

preds = {}
for fname in os.listdir(testDir):
    className = fname.split(".")[0]
    if className == "neut":
        className += "ral"
    cmd="diploSHIC predict clf.json clf.weights.h5 {0} {1} --simData".format(testDir+"/"+fname, predFile)
    os.system(cmd)
    preds[className] = Counter(np.loadtxt(predFile, skiprows=1, dtype=str)[:,0])
print(preds)

#here's the confusion matrix function
def makeConfusionMatrixHeatmap(data, title, trueClassOrderLs, predictedClassOrderLs, ax):
    data = np.array(data)
    data = normalize(data, axis=1, norm='l1')
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)

    for i in range(len(predictedClassOrderLs)):
        for j in reversed(range(len(trueClassOrderLs))):
            val = 100*data[j, i]
            if val > 50:
                c = '0.9'
            else:
                c = 'black'
            ax.text(i + 0.5, j + 0.5, '%.2f%%' % val, horizontalalignment='center', verticalalignment='center', color=c, fontsize=9)

    cbar = plt.colorbar(heatmap, cmap=plt.cm.Blues, ax=ax)
    cbar.set_label("Fraction of simulations assigned to class", rotation=270, labelpad=20, fontsize=11)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.axis('tight')
    ax.set_title(title)

    #labels
    ax.set_xticklabels(predictedClassOrderLs, minor=False, fontsize=9, rotation=45)
    ax.set_yticklabels(list(reversed(trueClassOrderLs)), minor=False, fontsize=9)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    
# convert our preds dictionary into a confusion matrix
classOrderLs=['hard', 'linkedHard', 'soft', 'linkedSoft', 'neutral']
counts = []
for trueClass in classOrderLs:
    currCounts = []
    for predClass in classOrderLs:
        currCounts.append(preds[trueClass][predClass])
    counts.append(currCounts)
counts.reverse()

#now do the plotting
fig,ax= plt.subplots(1,1)
makeConfusionMatrixHeatmap(counts, "Confusion matrix", classOrderLs, classOrderLs, ax)
plt.savefig(figFile, bbox_inches="tight")
