from math import ceil, floor
from scipy import stats 
import argparse 
import csv 
import numpy as np
import glob 
import os 

ARGS = argparse.ArgumentParser()
ARGS.add_argument('column',type=int)
ARGS.add_argument('-system1',nargs="+",required=True)
ARGS.add_argument('-system2',nargs="+",required=True)


if __name__ == "__main__":
    ARGS = ARGS.parse_args()

    system1_paths = ARGS.system1
    system2_paths = ARGS.system2

    average_readings = [[],[]]
    for i,system_paths in enumerate([system1_paths,system2_paths]):
        readings = []
        for s in system_paths:
            readings.append([])
            print(s)
            file = s if os.path.isfile(s) else  glob.glob(f'{s}/**/*results.csv',recursive=True)[0]
            with open(file ,'r')as f:
                reader = csv.reader(f)
                next(iter(reader))
                for r in reader:
                    if r[0] == 'AVG':
                        continue 
                    
                    readings[-1].append(float(r[ARGS.column]))

        for j in range(len(readings[0])):
            average_readings[i].append(
                sum([readings[y][j] for y in range(len(readings))]) / len(readings))

    print("Ranksums Wilcox (unpaired): ",stats.ranksums(average_readings[0],average_readings[1]))
    print("signed-rank Wilcox (paired): ",stats.wilcoxon(average_readings[0],average_readings[1]))
    print("System 1, mean, std")
    print(np.mean(average_readings[0]), np.std(average_readings[0]))
    print("System 2, mean, std")
    print(np.mean(average_readings[1]), np.std(average_readings[1]))

    import seaborn
    import matplotlib.pyplot as plt 

    diffs = [a-b for a,b in zip(average_readings[0],average_readings[1])]
    idx_min = np.argmin(diffs)
    print("Biggest difference: " + str(np.min(diffs)) + ", at line: " + str(idx_min+2) + ", s1:" + str(average_readings[0][idx_min]) + ", s2:" + str(average_readings[1][idx_min]))

    ax = seaborn.histplot(data=diffs,kde=True,color='blue',legend=True,bins=[x for x in np.linspace(floor(np.min(diffs))-1,ceil(np.max(diffs)+1),num=40)])
    plt.vlines(np.mean(diffs),0,ax.get_ylim()[1],colors='red',label='mean')
    plt.legend(labels=['sys1 - sys2','mean'])
    plt.show()
